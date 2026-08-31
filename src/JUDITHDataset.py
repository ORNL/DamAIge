import sys
from collections.abc import Iterable
from pathlib import Path
import re
import hashlib
import logging
import warnings
from functools import cached_property

import numpy as np
import pandas as pd
import openpyxl
import json

from pywt import data
from scipy.ndimage import find_objects
from skimage.io import imread
import tifffile

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.ticker import EngFormatter
engfmt = EngFormatter(unit="m", places=0, sep=" ")

import torch
from torch.utils.data import Dataset
# from torchvision import transforms

from transformers import AutoImageProcessor, AutoModel
from accelerate import Accelerator
from PIL import Image
from tqdm.auto import tqdm

# suppress warnings from tifffile about truncated images, since some JUDITH images are not perfectly saved
logging.getLogger('tifffile').setLevel(logging.CRITICAL)



def gpu_selector():
    # Select GPU with maximum memory

    dev_with_max_mem = 0
    max_mem = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"--- GPU {i} ---")
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        if max_mem<props.total_memory:
            max_mem = props.total_memory
            dev_with_max_mem = i
        print()
    torch.cuda.set_device(dev_with_max_mem)
    print('Selected GPU: ', torch.cuda.get_device_name(torch.cuda.current_device()))

    # set device
    return Accelerator().device


################################################################################
# Utility functions for labeling metadata fields with visual representations and interactive text boxes

# pixel bounds for data region (where the actual experiment images are located)
_data_pos = (slice(0,715), slice(0,1024))
# pixel bounds for scale metadata
_scale_pos    = (slice(715,735), slice(770,830))
_scalelen_pos = (slice(735,760), slice(770,1020))
# pixel bounds for detector metadata
_detector_pos = (slice(730,750), slice(370,540))


def patchify_fov(img, resolution, fov, stride_meters=None, stride_pixels=None, stride_ratio=None, *, flatten=False):
    '''Convert an image into patches of a given field of view (fov) in meters, with optional stride specifications.

    Parameters:
    - img: 2D numpy array representing the image
    - resolution: Physical size of a pixel in meters
    - fov: Field of view in meters (can be a single value or a tuple for height and width)
    - stride_meters: Optional stride in meters
    - stride_pixels: Optional stride in pixels
    - stride_ratio: Optional stride as a fraction of the patch size
    - flatten: Optional flag to return a 'list' of patches, i.e., flatten patch index dimension
    '''

    if not isinstance(img, np.ndarray) or img.ndim!=2:
        raise ValueError(f"img must be a 2D numpy array, got {type(img)} with shape {getattr(img, 'shape', None)}")

    h, w = img.shape

    #######################################################################
    # Convert fov to height/width in meters

    if isinstance(fov, (int, float)):
        fov_h = fov_w = float(fov)
    else:
        fov_h, fov_w = map(float, fov)

    if fov_h <= 0 or fov_w <= 0:
        raise ValueError(f"fov must be positive, got {fov!r}")

    # convert fov from meters to pixels using resolution metadata, ensuring at least 1 pixel
    ph = max(1, int(np.ceil(fov_h / resolution)))
    pw = max(1, int(np.ceil(fov_w / resolution)))

    #######################################################################
    # Determine stride in pixels

    if stride_meters is None and stride_pixels is None and stride_ratio is None:
        # Default stride is equal to patch size (non-overlapping patches)
        sh, sw = ph, pw
    elif stride_ratio is not None:
        # Stride is a fraction of the patch size
        if stride_ratio <= 0:
            raise ValueError("stride_ratio must be positive")
        sh = max(1, int(np.round(ph * float(stride_ratio))))
        sw = max(1, int(np.round(pw * float(stride_ratio))))
    elif stride_pixels is not None:
        # Stride is specified directly in pixels
        if isinstance(stride_pixels, int):
            stride_pixels = (stride_pixels, stride_pixels)
        elif stride_pixels is not None:
            stride_pixels = tuple(stride_pixels)
        sh, sw = stride_pixels
    else:
        # Stride is specified in physical units (microns), convert to pixels
        if isinstance(stride_meters, (int, float)):
            stride_meters = (float(stride_meters), float(stride_meters))
        elif stride_meters is not None:
            stride_meters = tuple(map(float, stride_meters))
        sh = max(1, int(np.round(float(stride_meters[0]) / resolution)))
        sw = max(1, int(np.round(float(stride_meters[1]) / resolution)))

    #######################################################################

    # skip images smaller than patch size
    if ph > h or pw > w:
        windows = np.empty((0, 0, ph, pw), dtype=img.dtype)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(img, (ph, pw))
        windows = windows[::sh, ::sw, :, :]

    # if enforce_view:
    #     if flatten:
    #         raise ValueError(
    #             "Cannot enforce a no-copy view with flattened patches. "
    #             "Use flatten=False to return a 4D sliding-window view."
    #         )
    #     return windows

    if flatten:
        image_patches = windows.reshape(-1, ph, pw)
        return image_patches

    # # Find maximum patch size across all images to pad smaller patches
    # max_ph = max(p.shape[1] for p in all_patches)
    # max_pw = max(p.shape[2] for p in all_patches)

    # resized = [sk_resize(patches, (patches.shape[0], max_ph, max_pw), order=1, mode="reflect", preserve_range=True, anti_aliasing=False) for patches in all_patches]

    return windows


def pad_to_shape(im, shape, val=255):
    '''Pad image to given shape with given values'''
    # Calculate padding for each dimension
    pad_width = []
    for i in range(len(im.shape)):
        diff = shape[i] - im.shape[i]
        if diff < 0:
            raise ValueError(f"Target shape dimension {i} is smaller than image dimension {i}. Consider cropping instead of padding.")
        pad_before = diff // 2
        pad_after = diff - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(im, pad_width, mode='constant', constant_values=val)


def extract_visual_metadata(im, pos):
    '''Extract metadata from image at the given position'''
    meta = im[pos]
    meta = (meta<meta.max()).astype(np.int8)
    bbox = find_objects(meta)[0]
    return 1 - pad_to_shape(meta[bbox], [sl.stop-sl.start for sl in pos], 0)


def label_visual_metadata(unique_items, is_float=False, title_prefix=""):
    """
    Generic interactive labeling function for metadata fields with visual representations.

    Parameters:
    - title_prefix: Prefix for plot titles and messages
    """
    unique_keys  = list(unique_items.keys())
    unique_items = list(unique_items.values())

    labeled_values  = {}

    # Create figure with subplots for images and text boxes
    n_items = len(unique_items)
    n_rows = 2

    fig = plt.figure(figsize=(n_items*3, 5))

    # Create subplots for images (top row)
    image_axes = []
    for i in range(n_items):
        ax = fig.add_subplot(n_rows, n_items, i+1)
        image_axes.append(ax)

    # Create subplots for string text boxes (bottom row)
    str_axes = []
    str_offset = n_items * 1 #(2 if has_values else 1)
    for i in range(n_items):
        ax = fig.add_subplot(n_rows, n_items, str_offset + i + 1)
        str_axes.append(ax)

    # Plot each unique item
    value_boxes = []
    str_boxes = []

    def make_str_callback(hash_val, box_index):
        def callback(text):
            if is_float:
                text = float(text)
            labeled_values[hash_val] = text
            print(f"{title_prefix} string '{text}' assigned to hash {hash_val}")
        return callback

    for i, (img_ax, item_img, item_key) in enumerate(zip(image_axes, unique_items, unique_keys)):
        # Plot the item image
        img_ax.imshow(item_img, cmap='gray')
        img_ax.axis('off')

        # Create text box for string representation
        str_ax = str_axes[i]
        str_ax.set_title(f'{title_prefix} string:')
        str_ax.axis('on')
        str_box = widgets.TextBox(str_ax, '', initial='')
        str_boxes.append(str_box)
        str_box.on_submit(make_str_callback(item_key, i))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    instructions = f'Enter {title_prefix.lower()} values and strings in text boxes below each image\nPress Enter to submit each value, then close window'
    fig.suptitle(instructions, fontsize=14)
    plt.show()

    return labeled_values


################################################################################
# Functions for reading the Excel test matrix and creating the dataset from images and metadata

# Grain size ranges (in microns) for each material, condition, and view (tv - top view, cs - cross section)
_grain_sizes = {
    'M192 - W-UHP':  {
        'as received':    {'tv': (53.2,66.7), 'cs': (48.3,169.5)},
        'recrystallised': {'tv': (76,125.9),  'cs': (74.2,146.9)}
    },
    'M193 - WVMW':   {
        'as received':    {'tv': (17.4,34.9), 'cs': (10.2,45.8)},
        'recrystallised': {'tv': (40.9,68),   'cs': (38.8,75)}
    },
    'M194 - WTa1':   {
        'as received':    {'tv': (9.5,17.7),  'cs': (5.6,16.4)},
        'recrystallised': {'tv': (18,27.2),   'cs': (14.5,23.9)}
    },
    'M195 - WTa5':   {
        'as received':    {'tv': (9.5,17.7),  'cs': (5.6,16.4)},
        'recrystallised': {'tv': (18,27.2),   'cs': (14.5,23.9)}
    },
    'M196 - pure W': {
        'as received':    {'tv': (40,121),    'cs': (42.3,264.0)},
        'recrystallised': {'tv': (50.6,78.7), 'cs': (37.7,68.1)}
    }
}



def read_excel_testmatrix_to_dict(file_path, sheet_name=None, header_row=1, start_row=None, end_row=None, start_col=2, end_col=None, ignore_symbols=None):
    """Read testmatrix in an Excel sheet and create a dictionary where each cell's value is the key.
    The value is a dict with the top header (from header_row) and left header (column 1).

    Rows are skipped entirely if they contain any symbol in ignore_symbols.

    Inputs:
        file_path: Path to Excel file
        sheet_name: Optional sheet name
        header_row: Row containing top headers
        start_row: First row of data (defaults to header_row + 1)
        end_row: Last row of data (defaults to sheet max row)
        start_col: First column of data (default = 2)
        end_col: Last column of data (defaults to sheet max column)
        ignore_symbols: list of symbols (strings) that cause row skipping

    Output:
        A dictionary mapping each cell value to a tuple (top_header, left_header)
    """

    # Load the workbook and select the worksheet
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    # Default values
    if start_row is None:
        start_row = header_row + 1
    if end_row is None:
        end_row = ws.max_row
    if end_col is None:
        end_col = ws.max_column
    if ignore_symbols is None:
        ignore_symbols = []  # no symbols ignored


    result = {}

    # ---- Read top headers ----
    base_temps = {}
    for col in range(start_col, end_col + 1):
        base_temps[col] = ws.cell(row=header_row, column=col).value

    # ---- Read data rows ----
    for row in range(start_row, end_row + 1):

        # --- Check if row should be ignored ---
        row_should_be_ignored = False
        if ignore_symbols:
            for col in range(1, end_col + 1):
                cell_value = ws.cell(row=row, column=col).value
                if cell_value is None:
                    continue
                cell_str = str(cell_value)
                if any(symbol in cell_str for symbol in ignore_symbols):
                    row_should_be_ignored = True
                    break

        if row_should_be_ignored:
            continue  # skip this row entirely

        # --- Get thermal flux for this row ---
        thermal_flux = ws.cell(row=row, column=1).value

        # --- Process data cells ---
        for col in range(start_col, end_col + 1):
            experiment_label = ws.cell(row=row, column=col).value
            if experiment_label is None:
                continue

            result[experiment_label] = (base_temps[col], thermal_flux)
    return result


def build_label2tempflux(file_path):
    """
    Build a mapping "experiment label" -> (base temperature, thermal flux) from the Excel test matrix.

    Reads sheets: 'Longitudinal', 'Transversal', 'Recrystalised' (header_row=2) and ignores rows containing the symbol 'RT' (Room Temperature).
    Composite keys like 'B3/B21/B36' are expanded so each sub-key maps to the same tuple.

    Inputs:
    - file_path: Path to Excel file containing the test matrix
    """

    # Read all relevant sheets and combine into a single mapping
    label_map = read_excel_testmatrix_to_dict(file_path,      sheet_name='Longitudinal',  header_row=2, start_col=2, end_col=9, ignore_symbols=['RT'])
    label_map.update(read_excel_testmatrix_to_dict(file_path, sheet_name='Transversal',   header_row=2, start_col=2, end_col=5, ignore_symbols=['RT']))
    label_map.update(read_excel_testmatrix_to_dict(file_path, sheet_name='Recrystalised', header_row=2, start_col=2, end_col=5, ignore_symbols=['RT']))

    # Expand composite keys like 'B3/B21/B36' -> {'B3':..., 'B21':..., 'B36':...}
    extra = {}
    for label, temp_flux in label_map.items():
        float_values = []
        for v in temp_flux:
            if '°C' in v:
                v = float(v.replace('°C',''))
            elif v=='RT':
                v = 20.0
            elif 'MW/m^2' in v:
                v = int(re.sub(r"^.*?\n", '', v, flags=re.DOTALL).replace(' MW/m^2',''))
            float_values.append(v)
        for sub in label.split('/'):
            extra[sub.strip()] = float_values
    label_map.update(extra)
    return label_map


def dataset_metadata_from_path(top_path, testmatrix_file="Testmatrix JUDITH 1.xlsx"):
    '''Generate dataset from images and metadata located under top_path.
    Assumed directory structure: top_path/<material>/<loadtype>/<label>_<id>.tif
    Metadata is extracted from the image filename and the Excel test matrix.

    On-the-fly tracking: unique scales and detectors are identified by their mean pixel value,
    and assigned sequential IDs as they are encountered.

    Inputs:
    - top_path: Path to the directory containing subdirectories for each material and load type
    - testmatrix_file: Excel file containing the test matrix with experiment labels and corresponding base temperatures and thermal fluxes

    Outputs:
    - Dictionary containing images and metadata, with scale_id and detector_id pointing to unique instances
    '''

    # Test matrix
    excel_file = Path(top_path, testmatrix_file)

    # Mapping "experiment label" -> (base temperature, thermal flux)
    label2tempflux = build_label2tempflux(excel_file)

    # images = []

    # Track unique scales and detectors on the fly
    unique_scales = {}      # key: hash of image, value: image
    unique_detectors = {}   # key: hash of image, value: image

    metadata = {'filename': [], 'material':[], 'loadtype':[], 'label':[], 'base_temp':[], 'flux':[], 'scale':[], 'detector':[], 'resolution':[], 'grain_size_tv':[], 'grain_size_cs':[]}
    for material_path in top_path.iterdir():
        if not material_path.is_dir(): continue
        for load_path in Path(material_path).iterdir():
            for img_path in Path(load_path).iterdir():
                if not '.tif' in str(img_path): continue
                metadata['filename'].append(str(img_path.resolve()))

                # original image
                img = imread(img_path).astype(int)
                # images.append(extract_visual_metadata(img, data_pos))

                # Extract scale and detector metadata
                scale_img    = extract_visual_metadata(img, _scale_pos)
                scalelen_img = extract_visual_metadata(img, _scalelen_pos)
                detector_img = extract_visual_metadata(img, _detector_pos)

                # Check if this scale has appeared before (using mean as key)
                scale_hash = hashlib.sha256(scale_img).hexdigest()
                if scale_hash not in unique_scales:
                    unique_scales[scale_hash] = scale_img
                metadata['scale'].append(scale_hash)

                # Check if this detector has appeared before (using mean as key)
                detector_hash = hashlib.sha256(detector_img).hexdigest()
                if detector_hash not in unique_detectors:
                    unique_detectors[detector_hash] = detector_img
                metadata['detector'].append(detector_hash)

                # material and load type from directory structure
                metadata['material'].append(material_path.parts[-1])
                metadata['loadtype'].append(load_path.parts[-1])

                # extract experiment label from filename (assumes format: <label>_<id>.tif)
                label = img_path.parts[-1].split("_")[0].split(".tif")[0]
                metadata['label'].append(label)

                # extract base temperature and flux from experiment label
                if label in label2tempflux.keys():
                    metadata['base_temp'].append(label2tempflux[label][0])
                    metadata['flux'].append(label2tempflux[label][1])
                else:
                    metadata['base_temp'].append(np.nan)
                    metadata['flux'].append(np.nan)

                # calculate length in pixels of the scale bar and save it in resolution field
                # (to be converted to actual resolution after scale labeling)
                bbox = find_objects(1-scalelen_img)[0]
                num_pix = bbox[1].stop - bbox[1].start
                metadata['resolution'].append(num_pix)

    # convert all lists to numpy arrays
    # images = np.stack(images, axis=0)

    # convert all lists to numpy arrays
    for k, v in metadata.items():
        metadata[k] = np.array(v)

    #########################################################################################################
    # label scales and detectors with interactive visual labeling function, using cached values if available

    label_file = Path(top_path, "labels.json")
    if label_file.is_file():
        with open(label_file, "r", encoding="utf-8") as f:
            scale_values, detector_strs = json.load(f)
    else:
        scale_values  = label_visual_metadata(unique_scales,    is_float=True,  title_prefix="Scale")
        detector_strs = label_visual_metadata(unique_detectors, is_float=False, title_prefix="Detector")
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump([scale_values, detector_strs], f, indent=2)
    metadata['scale']    = np.array([scale_values.get(h, np.nan) for h in metadata['scale']])
    metadata['detector'] = np.array([detector_strs.get(h, '')    for h in metadata['detector']])

    # calculate actual resolution using the labeled scale values and the pixel length of the scale bar
    metadata['resolution'] = metadata['scale'] / metadata['resolution']

    #########################################################################################################
    # label grain sizes for each material and load type using the predefined _grain_sizes dictionary

    metadata['grain_size_tv_min'] = np.full_like(metadata['resolution'], np.nan, dtype=float)
    metadata['grain_size_tv_max'] = np.full_like(metadata['resolution'], np.nan, dtype=float)
    metadata['grain_size_cs_min'] = np.full_like(metadata['resolution'], np.nan, dtype=float)
    metadata['grain_size_cs_max'] = np.full_like(metadata['resolution'], np.nan, dtype=float)
    for material in np.unique(metadata['material']):
        for loadtype in np.unique(metadata['loadtype']):
            grain_size = _grain_sizes.get(str(material), {}).get(str(loadtype), {'tv': (np.nan, np.nan), 'cs': (np.nan, np.nan)})
            mask = (metadata['material']==material) & (metadata['loadtype']==loadtype)
            metadata['grain_size_tv_min'][mask] = grain_size['tv'][0]
            metadata['grain_size_tv_max'][mask] = grain_size['tv'][1]
            metadata['grain_size_cs_min'][mask] = grain_size['cs'][0]
            metadata['grain_size_cs_max'][mask] = grain_size['cs'][1]

    return metadata


def metadata_filter(metadata, criteria=None, /, *, invert=False, return_mask=False):
    """Return a filtered Dataset matching metadata criteria.

    This creates a new `Dataset` containing only the rows (images) whose metadata fields match the provided criteria.

    Parameters
    criteria : dict[str, Any] | None
        Mapping from metadata field name to a match condition.
        If None, criteria are taken only from keyword arguments.

        Each value may be one of:
        - scalar: exact match, e.g. ``material='M196 - pure W'``.
            If the scalar is NaN, it matches missing values (NaN/None).
        - container (list/tuple/set/np.ndarray/range): membership match,
            e.g. ``loadtype={'as_received', 'recrystallised'}``.
            If the container includes NaN, missing values are included.
        - callable: predicate applied to the full metadata array for that
            field, returning a boolean mask of shape ``(len(self.images),)``.
            Example: ``base_temp=lambda a: a >= 800``.

    invert : bool, default=False
        If True, returns the complement of the selection (i.e. rows that do *not* match the criteria).

    return_mask : bool, default=False
        If True, returns a tuple ``(dataset, mask)`` where ``mask`` is a
        boolean array of shape ``(len(self.images),)`` indicating which
        rows were selected.

    **kwargs
        Convenience form for passing criteria without a dict.
        For example: ``ds.filter_by_metadata(material='M196 - pure W')``.
        If a field is provided in both ``criteria`` and ``kwargs``, a ``ValueError`` is raised.

    Returns
    dict[str, Any] | tuple[dict[str, Any], np.ndarray]
        Filtered metadata (and optionally the boolean selection mask).

    Notes
    - Filtering is ANDed across fields: all provided criteria must match.
    """

    merged = {}
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise TypeError("criteria must be a dict mapping field -> match, or None")
        merged.update(criteria)
    else:
        return metadata  # no criteria, return original

    n = len(metadata['filename'])
    mask = np.ones(n, dtype=bool)

    for field, desired in merged.items():
        if field not in metadata:
            raise KeyError(f"Unknown metadata field '{field}'. Available: {sorted(metadata.keys())}")
        values = metadata[field]
        values = np.asarray(values)

        if callable(desired):
            field_mask = np.asarray(desired(values), dtype=bool)
            if field_mask.shape != (n,):
                raise ValueError(f"Predicate for field '{field}' must return a boolean mask of shape ({n},), got {field_mask.shape}")
        else:
            is_container = isinstance(desired, (list, tuple, set, np.ndarray, range))
            desired_values = list(desired) if is_container else [desired]

            # Handle NaN membership explicitly (np.isin treats NaN as never-equal)
            has_nan = False
            non_nan = []
            for dv in desired_values:
                try:
                    if isinstance(dv, float) and np.isnan(dv):
                        has_nan = True
                    else:
                        non_nan.append(dv)
                except TypeError:
                    non_nan.append(dv)

            field_mask = np.zeros(n, dtype=bool)
            if non_nan:
                field_mask |= np.isin(values, non_nan)
            if has_nan:
                try:
                    field_mask |= pd.isna(values)
                except Exception:
                    # Fallback: best-effort NaN detection
                    field_mask |= np.asarray(values != values)

        mask &= field_mask

    if invert:
        mask = ~mask

    def _slice_meta(val):
        if isinstance(val, np.ndarray) and val.shape[:1] == (n,):
            return val[mask]
        if isinstance(val, list) and len(val) == n:
            return np.asarray(val)[mask]
        return val

    new_metadata = {k: _slice_meta(v) for k, v in metadata.items()}

    if return_mask:
        return new_metadata, mask
    return new_metadata


################################################################################
# PyTorch Dataset class for loading JUDITH images and metadata on demand, with optional filtering and transforms

class JUDITHDataset(Dataset):
    """PyTorch Dataset for JUDITH tungsten thermal-shock images.

    Loads images on-demand and returns normalized tensors with metadata.

    Args:
        data_path: Path to dataset directory containing 'metadata.npz' and image files
        transforms: Optional torchvision transforms to apply to images
        filter_criteria: Optional dict of metadata fields and values to filter the dataset (e.g. {'material': 'M192 - W-UHP'})
        normalize: Whether to normalize images to [0,1] range
        preload: Whether to preload all images into memory (may require more RAM)
        data_pos: Tuple of slices defining the region of the image containing the actual experiment data (default covers entire image)
    """

    def __init__(self, data_path=None, transforms=None, filter_criteria=None, normalize=True, preload=False, data_pos=None, remove_nan=True):
        # Capture all local arguments, excluding 'self'
        self.args = locals()
        del self.args['self']

        if data_path is None:
            data_path = Path(Path(__file__).resolve().parents[1], "data/JUDITH")
        if data_pos is None:
            data_pos = _data_pos
        self.data_path  = Path(data_path)
        self.metadata   = dict(np.load(Path(data_path, "metadata.npz"), allow_pickle=True))
        self.filter_criteria = filter_criteria
        self.transforms = transforms
        self.normalize  = normalize
        self.data_pos   = data_pos
        self.preload    = preload
        self.remove_nan = remove_nan

        # standardize load type naming
        self.metadata['loadtype'][self.metadata['loadtype']=='as received'] = 'longitudinal'

        # apply initial filter if specified
        self.metadata = metadata_filter(self.metadata, filter_criteria, invert=False, return_mask=False)

        # Remove NaN values if specified
        if remove_nan:
            self.metadata = metadata_filter(self.metadata, {'base_temp': lambda x: ~np.isnan(x)}, invert=False, return_mask=False)
            self.metadata = metadata_filter(self.metadata, {'flux': lambda x: ~np.isnan(x)}, invert=False, return_mask=False)

        # Preload images if specified (may require more RAM)
        if self.preload:
            img = tifffile.imread(self.metadata['filename'][0]).astype(np.float32)
            num_samples = len(self.metadata['filename'])
            self._images = np.empty((num_samples, *img.shape), dtype=np.float32)
            for i, fname in enumerate(self.metadata['filename']):
                self._images[i] = tifffile.imread(fname).astype(np.float32)

            self.unique_visual = {
                'scale':    np.array([extract_visual_metadata(im, _scale_pos)    for im in self._images]),
                'detector': np.array([extract_visual_metadata(im, _detector_pos) for im in self._images])
            }

        # cache unique values for each metadata field for quick access
        self.unique_meta = {key:np.unique(self.metadata[key]) for key in self.metadata if isinstance(self.metadata[key], np.ndarray)}


    def __len__(self):
        return len(self.metadata['filename'])


    def __getitem__(self, idx):
        # Load image
        if self.preload:
            img = self._images[idx]
        else:
            img_path = self.metadata['filename'][idx]
            img = tifffile.imread(img_path).astype(np.float32)

        # Extract data region
        img = img[self.data_pos[0], self.data_pos[1]]

        # Normalize
        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        # Prepare metadata dict for this sample
        sample_metadata = {
            'filename': self.metadata['filename'][idx],
            'material': self.metadata['material'][idx],
            'loadtype': self.metadata['loadtype'][idx],
            'label': self.metadata['label'][idx],
            'base_temp': torch.tensor(self.metadata['base_temp'][idx], dtype=torch.float32),
            'flux': torch.tensor(self.metadata['flux'][idx], dtype=torch.float32),
            'scale': torch.tensor(self.metadata['scale'][idx], dtype=torch.float32),
            'resolution': torch.tensor(self.metadata['resolution'][idx], dtype=torch.float32),
        }

        return img, sample_metadata


    #############################################################


    def filter_by_metadata(self, criteria=None, /, *, invert=False, return_mask=False, **kwargs):
        """Return a filtered Dataset matching metadata criteria.

        This creates a new `Dataset` containing only the rows (images) whose metadata fields match the provided criteria.

        Parameters
        criteria : dict[str, Any] | None
            Mapping from metadata field name to a match condition.
            If None, criteria are taken only from keyword arguments.

            Each value may be one of:
            - scalar: exact match, e.g. ``material='M196 - pure W'``.
              If the scalar is NaN, it matches missing values (NaN/None).
            - container (list/tuple/set/np.ndarray/range): membership match,
              e.g. ``loadtype={'as_received', 'recrystallised'}``.
              If the container includes NaN, missing values are included.
            - callable: predicate applied to the full metadata array for that
              field, returning a boolean mask of shape ``(len(self.images),)``.
              Example: ``base_temp=lambda a: a >= 800``.

        invert : bool, default=False
            If True, returns the complement of the selection (i.e. rows that do *not* match the criteria).

        return_mask : bool, default=False
            If True, returns a tuple ``(dataset, mask)`` where ``mask`` is a
            boolean array of shape ``(len(self.images),)`` indicating which
            rows were selected.

        **kwargs
            Convenience form for passing criteria without a dict.
            For example: ``ds.filter_by_metadata(material='M196 - pure W')``.
            If a field is provided in both ``criteria`` and ``kwargs``, a ``ValueError`` is raised.

        Returns
        Dataset | tuple[Dataset, np.ndarray]
            Filtered dataset (and optionally the boolean selection mask).

        Notes
        - Filtering is ANDed across fields: all provided criteria must match.

        Examples
        - ``ds2 = ds.filter_by_metadata(material='M196 - pure W', detector='QBSD')``
        - ``ds3 = ds.filter_by_metadata({'scale': [1e-4, 2e-4], 'flux': lambda a: a > 15})``
        """

        merged_criteria = self.filter_criteria.copy() if self.filter_criteria is not None else {}
        if criteria is not None:
            if not isinstance(criteria, dict):
                raise TypeError("criteria must be a dict mapping field -> match, or None")
            merged_criteria.update(criteria)

        # kwargs override / add, but duplicates are an error to avoid surprises
        for k, v in kwargs.items():
            if k in merged_criteria:
                raise ValueError(f"Duplicate criterion for field '{k}' provided in both criteria and kwargs")
            merged_criteria[k] = v

        _, mask = metadata_filter(self.metadata, merged_criteria, invert=invert, return_mask=True)

        args = self.args.copy()
        args.update({'filter_criteria': merged_criteria, 'preload': False})
        out = type(self)(**args)
        out.preload = self.preload  # keep the same preload setting as the original dataset
        # no need to read data again
        if self.preload:
            out._images = self._images[mask, :, :]

        if return_mask:
            return out, mask
        return out


    def add_metadata(self, field, values):
        """Add a new metadata field to the dataset.

        Parameters
        ----------
        field : str
            New metadata field name. Must not already exist.
        values : array-like
            Per-image values with length equal to ``len(self)``.
        """

        if field in self.metadata:
            raise KeyError(f"Metadata field '{field}' already exists")

        n = len(self)
        arr = np.asarray(values)

        if arr.shape == ():
            raise ValueError("values must be array-like with length equal to number of images")
        if arr.shape[0] != n:
            raise ValueError(f"values length {arr.shape[0]} does not match number of images {n}")

        self.metadata[field] = arr


    def remove_metadata(self, field):
        """Remove a metadata field from the dataset.

        Parameters
        ----------
        field : str
            Metadata field name to remove.
        """

        if field in self.metadata:
            self.metadata.pop(field, None)
            self.unique_meta.pop(field, None)
        else:
            warnings.warn(f"Metadata field '{field}' does not exist and cannot be removed")

    #############################################################

    @property
    def images(self):
        if self.preload:
            return self._images
        else:
            raise ValueError("Images are not preloaded. Set preload=True to enable this property.")

    @property
    def data(self):
        if self.preload:
            return self._images[:, self.data_pos[0], self.data_pos[1]]
        else:
            raise ValueError("Images are not preloaded. Set preload=True to enable this property.")

    @property
    def scales(self):
        return self.metadata['scale']

    @property
    def detectors(self):
        return self.metadata['detector']

    @property
    def resolutions(self):
        return self.metadata['resolution']

    @property
    def base_temps(self):
        return self.metadata['base_temp']

    @property
    def fluxes(self):
        return self.metadata['flux']

    @property
    def materials(self):
        return self.metadata['material']

    @property
    def loadtypes(self):
        return self.metadata['loadtype']

    @property
    def labels(self):
        return self.metadata['label']

    @property
    def filenames(self):
        return self.metadata['filename']


    #############################################################

    def summary_table(self):
        """Generate a summary table of dataset statistics"""

        materials = list(self.unique_meta['material'])
        scales    = list(self.unique_meta['scale'][::-1])

        material_masks = [self.metadata['material'] == m for m in materials]
        scale_masks    = [self.metadata['scale'] == s for s in scales]

        def _build_joined_counts(split_key, split_values):
            """Build row-wise and scale-wise joined count strings for a metadata split."""
            split_masks = [self.metadata[split_key] == value for value in split_values]

            # Rows: [all materials, material_1, material_2, ...]
            total_by_row = [[] for _ in range(len(materials) + 1)]
            # Shape: (rows, scales)
            scale_by_row = [[[] for _ in scales] for _ in range(len(materials) + 1)]

            for split_mask in split_masks:
                total_by_row[0].append(str(np.sum(split_mask)))
                for i, material_mask in enumerate(material_masks, start=1):
                    total_by_row[i].append(str(np.sum(split_mask & material_mask)))

                for j, scale_mask in enumerate(scale_masks):
                    scale_by_row[0][j].append(str(np.sum(split_mask & scale_mask)))
                    for i, material_mask in enumerate(material_masks, start=1):
                        scale_by_row[i][j].append(
                            str(np.sum(split_mask & scale_mask & material_mask))
                        )

            total_joined = [" / ".join(parts) for parts in total_by_row]
            scale_joined = np.array(
                [[" / ".join(parts) for parts in row] for row in scale_by_row],
                dtype=object,
            )
            return total_joined, scale_joined

        # Total number of images for each material
        data_stat = pd.DataFrame(
            data=[len(self)] + [np.sum(self.metadata['material'] == m) for m in materials],
            index=['all'] + materials,
            columns=pd.MultiIndex.from_tuples([(' ', 'total')]),
        )

        # Total number of experiments for each material
        num_exp = [
            len(np.unique(self.metadata['label'][self.metadata['material'] == m]))
            for m in materials
        ]
        data_stat[(" ", '# experiments')] = [sum(num_exp)] + num_exp

        # Total number of images for each load type
        loadtypes = list(self.unique_meta['loadtype'])
        loadtype_totals, loadtype_scales = _build_joined_counts('loadtype', loadtypes)
        loadtype_header = " / ".join(str(loadtype) for loadtype in loadtypes)
        data_stat[(loadtype_header, 'total')] = loadtype_totals
        for j, scale in enumerate(scales):
            data_stat[(loadtype_header, engfmt(scale))] = loadtype_scales[:, j]

        # Total number of images for each load type per detector
        detectors = list(self.unique_meta['detector'])
        detector_totals, detector_scales = _build_joined_counts('detector', detectors)
        detector_header = " / ".join(str(detector) for detector in detectors)
        data_stat[(detector_header, 'total')] = detector_totals
        for j, scale in enumerate(scales):
            data_stat[(detector_header, engfmt(scale))] = detector_scales[:, j]

        return data_stat


    def plot_test_matrix(self, filter_criteria=None, field='crack_density', cmap='viridis', what_to_show='heatmaps'):
        '''Plot a test matrix of the dataset, showing the specified metadata field as heatmaps or circle maps.

        Parameters:
        - filter_criteria: dict of metadata fields to filter the dataset before plotting (e.g. {'material': 'M196 - pure W'})
        - field: metadata field to visualize (default: 'crack_density')
        - cmap: colormap for heatmaps (default: 'viridis')
        - what_to_show: 'heatmaps', 'circle maps', or 'label' to determine the visualization style
        '''
        from matplotlib.patches import Rectangle, Circle

        filtered_dataset = self.filter_by_metadata(filter_criteria, invert=False, return_mask=False)

        # Build a test matrix of the specified field for each combination of load type, material, flux, and base temperature
        test_matrix = {}
        for load_type in self.unique_meta['loadtype']:
            for material in self.unique_meta['material']:
                for flux in self.unique_meta['flux']:
                    for base_temp in self.unique_meta['base_temp']:
                        testmat_dataset = filtered_dataset.filter_by_metadata(material=material, loadtype=load_type, flux=flux, base_temp=base_temp)
                        if len(testmat_dataset)>0:
                            if field=='num images':
                                test_matrix[(load_type, material, flux, base_temp)] = len(testmat_dataset)
                            else:
                                test_matrix[(load_type, material, flux, base_temp)] = np.nanmean(testmat_dataset.metadata[field])
        vmin = 0
        vmax = np.max(list(test_matrix.values()))

        # what_to_show = 'heatmaps' # 'heatmaps', 'circle maps', 'label'
        # cmap = 'viridis' # 'binary', 'gray', 'hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

        num_cols = len(self.unique_meta['material'])
        num_rows = len(self.unique_meta['loadtype'])
        if what_to_show in ['heatmaps', 'label']:
            width  = 3.7
            height = 4.0
        else:
            width  = 3.4
            height = 4.0

        fig, axes = plt.subplots(num_rows, num_cols,
            figsize=(width*num_cols, height*num_rows),
            constrained_layout=True,
            gridspec_kw={
                "wspace": 0.03,   # reduce horizontal spacing
                "hspace": 0.05    # keep vertical spacing
            })

        for axj, (load_type, axrow) in enumerate(zip(self.unique_meta['loadtype'], axes)):
            for axi, (material, ax) in enumerate(zip(self.unique_meta['material'], axrow)):
                arr = np.full((len(filtered_dataset.unique_meta['flux']), len(filtered_dataset.unique_meta['base_temp'])), np.nan)

                for i, flux in enumerate(filtered_dataset.unique_meta['flux']):
                    for j, base_temp in enumerate(filtered_dataset.unique_meta['base_temp']):
                        arr[i, j] = test_matrix.get((load_type, material, flux, base_temp), np.nan)

                nrows, ncols = arr.shape

                if what_to_show == 'heatmaps':
                    last_im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='equal')
                    # ---- axis labeling using GLOBAL grid ----
                    ax.set_xticks(np.arange(len(filtered_dataset.unique_meta['base_temp'])))
                    ax.set_xticklabels([f'{t:g}' for t in filtered_dataset.unique_meta['base_temp']], rotation=45)
                    ax.set_yticks(np.arange(len(filtered_dataset.unique_meta['flux'])))
                    ax.set_yticklabels([f'{f:g}' for f in filtered_dataset.unique_meta['flux']])

                    # Optional pixel grid
                    ax.set_xticks(np.arange(-0.5, len(filtered_dataset.unique_meta['base_temp']), 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, len(filtered_dataset.unique_meta['flux']), 1), minor=True)
                    ax.grid(which='minor', color='k', linewidth=0.3)
                    ax.tick_params(which='minor', bottom=False, left=False)
                elif what_to_show == 'label':
                    last_im = ax.imshow(label, cmap='binary', vmin=0, vmax=10, origin='lower', aspect='equal')
                    # ---- axis labeling using GLOBAL grid ----
                    ax.set_xticks(np.arange(len(filtered_dataset.unique_meta['base_temp'])))
                    ax.set_xticklabels([f'{t:g}' for t in filtered_dataset.unique_meta['base_temp']], rotation=45)
                    ax.set_yticks(np.arange(len(filtered_dataset.unique_meta['flux'])))
                    ax.set_yticklabels([f'{f:g}' for f in filtered_dataset.unique_meta['flux']])

                    # Optional pixel grid
                    ax.set_xticks(np.arange(-0.5, len(filtered_dataset.unique_meta['base_temp']), 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, len(filtered_dataset.unique_meta['flux']), 1), minor=True)
                    ax.grid(which='minor', color='k', linewidth=0.3)
                    ax.tick_params(which='minor', bottom=False, left=False)
                elif what_to_show == 'circle maps':
                    # Draw white boxes for each cell
                    for i in range(nrows):
                        for j in range(ncols):
                            val = arr[i, j]

                            if not np.isnan(val):
                                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='gray', alpha=0.2, edgecolor='k', linewidth=0.3))
                            else:
                                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', alpha=0.2, edgecolor='k', linewidth=0.3))
                                continue

                            # Map value to circle radius
                            frac = (val - vmin) / (vmax - vmin)
                            frac = np.clip(frac, 0, 1)
                            radius = 0.5 * frac

                            if radius > 0:
                                ax.add_patch(Circle((j, i), radius=radius, facecolor='black', edgecolor='none'))

                if axi == 0:
                    ax.set_ylabel(f"{load_type}\n\n"+r"Flux, $MW/m^2$", fontsize=12)
                if axj == 0:
                    ax.set_title(f"{material}", fontsize=12)
                if axj == num_rows - 1:
                    ax.set_xlabel(f"Base temperature, °C", fontsize=12)

                # disable ticks for all but the bottom row and leftmost column
                if axj == num_rows - 1:
                    ax.set_xticks(np.arange(len(self.unique_meta['base_temp'])))
                    ax.set_xticklabels([f'{t:g}' for t in self.unique_meta['base_temp']], rotation=45)
                else:
                    ax.set_xticks([])
                if axi == 0:
                    ax.set_yticks(np.arange(len(self.unique_meta['flux'])))
                    ax.set_yticklabels([f'{f:g}' for f in self.unique_meta['flux']])
                else:
                    ax.set_yticks([])


                # Match cell boundaries exactly
                ax.set_xlim(-0.5, ncols - 0.5)
                ax.set_ylim(-0.5, nrows - 0.5)
                ax.set_aspect('equal')

        field_str = ''
        if field == 'crack_density':
            field_str = 'Crack density (%)'
        elif field == 'num images':
            field_str = 'Number of images'
        else:
            field_str = field

        if what_to_show in ['heatmaps', 'label']:
            # ---------- shared colorbar ----------
            cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=1, pad=0.02)
            # cbar.ax.tick_params(labelsize=12)
            # cbar.set_label(f'{field_str}', fontsize=14)

        if np.unique(filtered_dataset.metadata['detector']).size == 1:
            detector = f"{filtered_dataset.metadata['detector'][0]}"
        else:
            detector = 'multiple detectors'
        if np.unique(filtered_dataset.metadata['scale']).size == 1:
            scale = filtered_dataset.metadata['scale'][0]
            plt.suptitle(f"{field_str} across materials and conditions, {engfmt(scale)} scale, {detector} detector(s)", fontsize=20, x=0.5, y=1.05)
        else:
            plt.suptitle(f"{field_str} across materials and conditions, multiple scales, {detector}", fontsize=20, x=0.5, y=1.05)
        plt.show()



class JUDITHPatchDataset(JUDITHDataset):
    """PyTorch Dataset for JUDITH tungsten thermal-shock images, returning patches of a specified field of view (FOV) with a given stride ratio.
    Data and metadata are kept at the image level, and converted to patches only at __getitem__
    """

    def __init__(self, fov, stride_ratio, data_path=None, transforms=None, filter_criteria=None, normalize=True, preload=False, data_pos=None, remove_nan=True):
        super().__init__(data_path=data_path, transforms=transforms, filter_criteria=filter_criteria, normalize=normalize, preload=preload, data_pos=data_pos, remove_nan=remove_nan)
        # Capture all local arguments, excluding 'self'
        self.args = locals()
        del self.args['self']
        del self.args['__class__']

        self.fov = fov
        self.stride_ratio = stride_ratio

        # Precompute the total number of patches across all images in the dataset
        num_patches = 0
        self.num_patches_per_image = []
        # self.patches = []
        for i in range(super().__len__()):
            img, meta = super().__getitem__(i)
            img = img.squeeze(0).numpy()  # Get the image as a numpy array
            patches = patchify_fov(img, meta['resolution'].numpy(), self.fov, stride_meters=None, stride_pixels=None, stride_ratio=self.stride_ratio, flatten=False)
            self.num_patches_per_image.append(patches.shape[0] * patches.shape[1])
            # self.patches.append(patches)
            num_patches += self.num_patches_per_image[-1]
        self.cumulative_patches = np.cumsum(self.num_patches_per_image)
        self._cached_num_patches = num_patches


    def idx_to_img_idx(self, idx):
        """Convert a global patch index to the corresponding image index and local patch index within that image."""
        if idx < 0 or idx >= self._cached_num_patches:
            raise IndexError(f"Index {idx} is out of bounds for total patches {self._cached_num_patches}")

        img_idx = np.searchsorted(self.cumulative_patches, idx, side='right')
        local_patch_idx = idx - (self.cumulative_patches[img_idx - 1] if img_idx > 0 else 0)
        return img_idx, local_patch_idx


    def img_idx_to_patches(self, img_idx, flatten=True):
        # Get the image as a numpy array
        img = super().__getitem__(img_idx)[0].squeeze(0).numpy()
        return patchify_fov(img, self.metadata['resolution'][img_idx], self.fov, stride_meters=None, stride_pixels=None, stride_ratio=self.stride_ratio, flatten=flatten)


    def __len__(self):
        return self._cached_num_patches


    def __getitem__(self, idx):
        '''Get a patch from the dataset based on the global patch index.
        Returns the patch and its associated metadata.
        '''
        flip = False
        if idx >= self._cached_num_patches:
            flip = True
            idx = idx - self._cached_num_patches

        img_idx, local_patch_idx = self.idx_to_img_idx(idx)
        img = super().__getitem__(img_idx)[0].squeeze(0).numpy()  # Get the image as a numpy array

        if flip:
            img = np.flip(img, axis=(0, 1)).copy() # copy to avoid negative strides in memory layout
        patches = patchify_fov(img, self.metadata['resolution'][img_idx], self.fov, stride_meters=None, stride_pixels=None, stride_ratio=self.stride_ratio, flatten=False)

        local_patch_idx_i = local_patch_idx // patches.shape[1]
        local_patch_idx_j = local_patch_idx % patches.shape[1]

        patch = torch.from_numpy(patches[local_patch_idx_i, local_patch_idx_j].copy()).unsqueeze(0)

        # Prepare metadata dict for this sample
        sample_metadata = {
            'filename': self.metadata['filename'][img_idx],
            'material': self.metadata['material'][img_idx],
            'loadtype': self.metadata['loadtype'][img_idx],
            'label': self.metadata['label'][img_idx],
            'base_temp': torch.tensor(self.metadata['base_temp'][img_idx], dtype=torch.float32),
            'flux': torch.tensor(self.metadata['flux'][img_idx], dtype=torch.float32),
            'scale': torch.tensor(self.metadata['scale'][img_idx], dtype=torch.float32),
            'resolution': torch.tensor(self.metadata['resolution'][img_idx], dtype=torch.float32),
        }

        return patch, sample_metadata


    def plot_patch_overlays(self, img_idx, pattern='all', show=True, ax=None):
        '''Plot the original image with overlaid patch boundaries for a given image index.
        Useful for visualizing how patches are extracted from the image.
        '''
        from matplotlib.patches import Rectangle

        if ax is None:
            ax = plt.gca()

        if img_idx >= 2*super().__len__():
            raise IndexError(f"Image index {img_idx} is out of bounds for total images {2*super().__len__()} (including flipped images)")

        flip = False
        if img_idx >= super().__len__():
            flip = True
            img_idx = img_idx - super().__len__()

        img = super().__getitem__(img_idx)[0].squeeze(0).numpy()  # Get the image as a numpy array
        if flip:
            img = np.flip(img, axis=(0, 1)).copy() # copy to avoid negative strides in memory layout

        ax.imshow(img, cmap='gray')

        # Calculate patch size in pixels
        resolution = self.metadata["resolution"][img_idx]
        patch_size_pixels = int(np.ceil(self.fov / resolution))
        stride_pixels = max(1, int(np.round(patch_size_pixels * self.stride_ratio)))

        if pattern == 'all':
            for i in range(0, img.shape[0] - patch_size_pixels + 1, stride_pixels):
                for j in range(0, img.shape[1] - patch_size_pixels + 1, stride_pixels):
                    rect = Rectangle((j, i), patch_size_pixels, patch_size_pixels, linewidth=1, edgecolor='red', facecolor='none', linestyle='-')
                    ax.add_patch(rect)
        elif pattern == 'diagonal':
            for i in range(0, img.shape[0] - patch_size_pixels + 1, stride_pixels):
                j = i  # diagonal
                if j < img.shape[1] - patch_size_pixels + 1:
                    rect = Rectangle((j, i), patch_size_pixels, patch_size_pixels, linewidth=1, edgecolor='red', facecolor='none', linestyle='-')
                    ax.add_patch(rect)

        # Add label for the patch
        ax.text(20 + patch_size_pixels/2, 20 + patch_size_pixels/2, f'{engfmt(self.fov)}',
                    color='red', fontsize=14, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        title_str = f'Resolution: {engfmt(resolution)}, scale: {engfmt(self.metadata["scale"][img_idx])}'
        # if plot_patch:
        #      title_str += f', full patches: {num_patches_x}x{num_patches_y}'
        ax.set_title(title_str, fontsize=14)
        ax.axis('off')

        if show:
            plt.show()



class JUDITHPatchEncodingDataset(JUDITHPatchDataset):
    """PyTorch Dataset for JUDITH tungsten thermal-shock images, returning encodings of patches of a specified field of view (FOV) with a given stride ratio.
    """

    def __init__(self, fov, stride_ratio, data_path=None, transforms=None, filter_criteria=None, normalize=True, preload=False, data_pos=None, remove_nan=True):
        super().__init__(fov=fov, stride_ratio=stride_ratio, data_path=data_path, transforms=transforms, filter_criteria=filter_criteria, normalize=normalize, preload=preload, data_pos=data_pos, remove_nan=remove_nan)
        # Capture all local arguments, excluding 'self'
        self.args = locals()
        del self.args['self']
        del self.args['__class__']

        self.encoder_name = 'facebook/dinov2-base'
        self.processor = AutoImageProcessor.from_pretrained(self.encoder_name)
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        self.encoder.eval()


    def __getitem__(self, idx):
        '''Get encoding of patch from the dataset based on the global patch index. Returns the encoding and its associated metadata.
        '''
        patch, meta = super().__getitem__(idx)
        patch = patch.expand(3,-1,-1) #.squeeze(0).numpy()  # Get the patch as a numpy array

        # Encode the patch
        with torch.no_grad():
            inputs = self.processor(patch, return_tensors="pt")
            inputs = {k: v for k, v in inputs.items()}
            encoding = self.encoder(**inputs).last_hidden_state.mean(dim=1).squeeze(0)

        return encoding, meta



################################################################################


def evaluate_crack_densities(dataset):
    #############################################################
    # Loop through all material-loadtype combinations and compute crack densities for each flux and base_temp
    print("Extracting crack densities...")

    scale = 1.e-3
    detector = 'QBSD'

    # Compute crack densities for each material-loadtype-flux-base_temp combination
    crack_densities_dict = {}
    crack_dataset = dataset.filter_by_metadata(scale=scale, detector=detector)
    print(len(crack_dataset), "images for scale", scale, "and detector", detector)
    for load_typei in dataset.unique_meta['loadtype']:
        for materiali in dataset.unique_meta['material']:
            material_load_dataset = crack_dataset.filter_by_metadata(material=materiali, loadtype=load_typei)

            print(f'{len(material_load_dataset):4d} images for material {materiali:13s} and load type {load_typei}')

            # Loop through flux and base_temp combinations for this material-loadtype and compute crack density for each
            dnst   = np.zeros((len(dataset.unique_meta['flux']), len(dataset.unique_meta['base_temp'])))
            numdat = np.zeros((len(dataset.unique_meta['flux']), len(dataset.unique_meta['base_temp'])))
            dnsts  = []
            for i, fluxi in enumerate(dataset.unique_meta['flux']):
                for j, base_tempi in enumerate(dataset.unique_meta['base_temp']):
                    flux_temp_dataset = material_load_dataset.filter_by_metadata(flux=fluxi, base_temp=base_tempi)
                    if len(flux_temp_dataset.data) > 0:
                        for dat in flux_temp_dataset.data:
                            cracks = find_cracks(dat)
                            den = np.mean(cracks[10:-10,10:-10]) # avoid borders which can have artifacts
                            dnst[i,j] += den
                            numdat[i,j] += 1
                        dnsts.append(dnst)
                        crack_densities_dict[(materiali,load_typei,fluxi,base_tempi)] = dnst[i,j] / numdat[i,j] * 100

    # Update dataset metadata with crack densities
    if 'crack_density' in dataset.metadata:
        dataset.remove_metadata('crack_density')
    dataset.add_metadata('crack_density', [np.nan]*len(dataset))
    for (materiali,load_typei,fluxi,base_tempi), den in crack_densities_dict.items():
        mask = (
            (dataset.metadata['material'] == materiali) &
            (dataset.metadata['loadtype'] == load_typei) &
            (dataset.metadata['flux'] == fluxi) &
            (dataset.metadata['base_temp'] == base_tempi)
        )
        dataset.metadata['crack_density'][mask] = den

    return dataset


def compute_encodings(dataset, fovs=[20,50,100,200], batch_size=64, detector='QBSD'):
    '''Compute encodings for each image in the dataset using a pretrained model (DINOv2).
    The encodings are computed for different fields of view (FOVs) and stored in the dataset's metadata.
    '''
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from accelerate import Accelerator
    from PIL import Image
    from tqdm.auto import tqdm

    print("Computing encodings for each image using a pretrained model...")

    ##################################
    # Select GPU with maximum memory
    dev_with_max_mem = 0
    max_mem = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"--- GPU {i} ---")
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"Compute Capability: {props.major}.{props.minor}")
        if max_mem<props.total_memory:
            max_mem = props.total_memory
            dev_with_max_mem = i
        print()
    torch.cuda.set_device(dev_with_max_mem)
    print('Selected GPU: ', torch.cuda.get_device_name(torch.cuda.current_device()))

    # set device
    device = Accelerator().device
    ##################################

    # Texture dataset uses the specified detector and resolution range (3e-8 to 5e-6)
    texture_dataset, texture_data_mask = dataset.filter_by_metadata(detector=detector, resolution=lambda a: np.logical_and(a>3e-8, a<5e-6), return_mask=True)

    ##################################
    # Load pretrained DINOv2 model and processor

    model_name = 'facebook/dinov2-base'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    patch_size_dino = model.config.patch_size
    hidden_dim_dino = model.config.hidden_size
    print(f"DINO (patch size, hidden dim): ({patch_size_dino}, {hidden_dim_dino})")

    ##################################

    encodings = {}

    # Compute encodings for each FOV
    for fov in fovs:
        print(f"Processing fov {fov} microns...")

        data_encodings = np.full((len(dataset), hidden_dim_dino), np.nan)
        texture_data_encodings = np.full((len(texture_dataset), hidden_dim_dino), np.nan)

        # Loop through each unique experiment label in the texture dataset and compute encodings for images with that label
        for lbl_i, lbl in enumerate(tqdm(texture_dataset.unique_meta['label'], desc=f"FOV ({fov}um)", leave=False)):
            experiment_dataset, experiment_data_mask = texture_dataset.filter_by_metadata(label=lbl, return_mask=True)
            lbl_data_encodings = np.full((len(experiment_dataset), hidden_dim_dino), np.nan)

            all_patches = [patchify_fov(data, resolution, fov*1e-6, stride_ratio=1.0, flatten=True) for data,resolution in zip(experiment_dataset.data, experiment_dataset.metadata['resolution'])]

            # Loop through each image in the experiment dataset and compute encodings for its patches
            for imgi, img_patches in tqdm(enumerate(all_patches), total=len(all_patches), desc=f"   Label ({lbl})", leave=False):
                if len(img_patches)>0:
                    num_batches = int(np.ceil(len(img_patches) / batch_size))
                    with torch.no_grad():
                        img_patches_dino_embeddings = []
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * batch_size
                            end_idx = min((batch_idx + 1) * batch_size, len(img_patches))

                            # Get patches for this batch and reshape to 2D
                            batch_patches_2d = img_patches[start_idx:end_idx]

                            # Convert to 3-channel images (stack grayscale 3 times)
                            batch_images = np.stack([batch_patches_2d] * 3, axis=-1)

                            # Convert to PIL Images for processor
                            pil_images = [Image.fromarray(img.astype(np.uint8)) for img in batch_images]

                            # Process through DINO
                            inputs  = processor(images=pil_images, return_tensors="pt").to(device)
                            outputs = model(**inputs)

                            last_hidden_states = outputs.last_hidden_state  # [batch_size, num_patches + 1, hidden_dim]

                            # Extract patch embeddings (skip CLS token at index 0)
                            # patch_embeddings = last_hidden_states[:, 1:, :]  # Skip CLS token

                            # # Average pool across spatial dimensions to get single embedding per patch
                            # patch_embeddings_pooled = patch_embeddings.mean(dim=1)  # [batch_size, hidden_dim]

                            # CLS token embedding (first token) can be used as a representation of the entire image
                            patch_embeddings = last_hidden_states[:, 0, :] # [batch_size, hidden_dim]

                            img_patches_dino_embeddings.append(patch_embeddings.cpu().numpy())

                            # if (batch_idx + 1) % 10 == 0:
                            #     print(f"  Processed {end_idx}/{len(img_patches)} patches")

                        # average the encodings for all patches of this image to get a single encoding for that image
                        lbl_data_encodings[imgi, :] = np.mean(np.vstack(img_patches_dino_embeddings), axis=0)
            # average the encodings for all images with the same label to get a single encoding for that label/experiment
            texture_data_encodings[experiment_data_mask, :] = lbl_data_encodings.mean(axis=0, keepdims=True)

        data_encodings[texture_data_mask, :] = texture_data_encodings

        encodings[f'dino2_encoding_fov{fov}'] = data_encodings

    # np.savez(Path(top_path, "encodings.npz"), **encodings)
    return encodings


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from crack_identification import find_cracks

    top_path = Path(Path(__file__).resolve().parents[1], "data/JUDITH")
    print(top_path)

    #############################################################
    print("Generating dataset metadata from images and Excel test matrix...")

    dataset_metadata = dataset_metadata_from_path(top_path)
    np.savez(Path(top_path, "metadata.npz"), **dataset_metadata)

    dataset = JUDITHDataset(preload=True, data_path=top_path, remove_nan=False)

    #############################################################
    # Loop through all material-loadtype combinations and compute crack densities for each flux and base_temp

    dataset = evaluate_crack_densities(dataset)
    np.savez(Path(top_path, "metadata.npz"), **dataset.metadata)


    #############################################################
    # compute encodings

    encodings = compute_encodings(dataset, fovs=[20,50,100,200], batch_size=64, detector='QBSD')
    np.savez(Path(top_path, "encodings.npz"), **encodings)
