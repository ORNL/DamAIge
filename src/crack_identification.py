import numpy as np

from skimage.restoration import denoise_tv_bregman
from skimage import exposure
from skimage.exposure import histogram as im_hist
from skimage.filters import unsharp_mask, gaussian as im_gaussian, median as im_median
from skimage import filters, morphology
from skimage.feature import structure_tensor, structure_tensor_eigenvalues



def enhance_cracks(img, outlier_percentiles=(0, 100), blur_sigma=10, tv_weight=5, ridge_method='sato', ridge_sigmas=range(1, 4)):
    # Remove outliers by rescaling intensities based on 1st and 99th percentiles
    p1, p2 = np.percentile(img, outlier_percentiles)
    img = exposure.rescale_intensity(img, in_range=(p1, p2), out_range=(0, 1))

    # img = exposure.equalize_adapthist(img)

    # flatten background - estimate slow-varying background with a large Gaussian blur
    # background = filters.gaussian(img, sigma=blur_sigma)
    # img = exposure.rescale_intensity(img-background, out_range=(0, 1))

    # img = denoise_tv_bregman(img, weight=tv_weight)

    # ridge enhancement - Sato/Frangi/Meijering can all work
    if ridge_method == 'sato':
        img = filters.sato(img, sigmas=ridge_sigmas, black_ridges=True)
    elif ridge_method == 'frangi':
        img = filters.frangi(img, sigmas=ridge_sigmas, black_ridges=True)
    elif ridge_method == 'meijering':
        img = filters.meijering(img, sigmas=ridge_sigmas, black_ridges=True)
    else:
        raise ValueError("Unknown ridge enhancement method")
    img = exposure.rescale_intensity(img, out_range=(0, 1))

    # remove completely dark pixels (which can cause issues for some thresholding methods)
    # img = exposure.rescale_intensity(img, out_range=(5/256, 1))
    # p10 = np.percentile(img, 50)
    # img = exposure.rescale_intensity(img, in_range=(p10, 100), out_range=(0, 1))

    # # morphological closing to connect nearby ridges into more complete cracks
    # h,w = img.shape
    # kernel = np.ones((min(h,w)//70,min(h,w)//70)) # adaptive kernel size based on image dimensions
    # img = morphology.closing(img, kernel)

    for _ in range(0):
        # denoise small specks that could interfere with thresholding and skeletonization
        img = denoise_tv_bregman(img, weight=tv_weight)
        # img = denoise_wavelet(img)
        # img = denoise_bilateral(img, sigma_color=None, sigma_spatial=5)
        # img = denoise_nl_means(img, patch_size=21, patch_distance=100)
        img = exposure.rescale_intensity(img, out_range=(0, 1))

        # histogram equalization to further enhance contrast before thresholding
        img = exposure.equalize_adapthist(img)
        img = exposure.rescale_intensity(img, out_range=(0, 1))

        # img = unsharp_mask(img, radius=2, amount=3, preserve_range=True)
        # img = exposure.rescale_intensity(img, out_range=(0, 1))

    # histogram equalization to further enhance contrast before thresholding
    # img = exposure.equalize_adapthist(img)
    # img = exposure.rescale_intensity(img, out_range=(0, 1))

    img = im_gaussian(img, sigma=1)
    img = unsharp_mask(img, radius=2, amount=2, preserve_range=True)
    img = exposure.rescale_intensity(img, out_range=(0, 1))

    # img = im_median(img, footprint=np.ones((10,10)))
    # img = exposure.equalize_adapthist(img)
    # img = exposure.rescale_intensity(img, out_range=(0, 1))

    return img


def crack_detector(im, method='triangle', min_coherence=0.8, quantile=0.95, black_cracks=True):
    kernel = np.ones((5,5))
    # h,w = im.shape
    # kernel = np.ones((min(h,w)//50,min(h,w)//50)) # adaptive kernel size based on image dimensions

    # th_vals = filters.threshold_multiotsu(im, classes=3)
    # im = exposure.rescale_intensity(im, out_range=(th_vals[0], th_vals[1]))

    # Compute global threshold
    if method=='yen':
        th_val = filters.threshold_yen(im)
    elif method=='otsu':
        th_val = filters.threshold_otsu(im)
    elif method=='multiotsu':
        th_vals = filters.threshold_multiotsu(im, classes=3)
        # find smallest threshold that captures at least 10% of pixels as cracks
        th_val = 0
        for th in th_vals:
            if (im<th).sum()>=0.1*im.size:
                break
            th_val = th
    elif method=='minimum':
        th_val = filters.threshold_minimum(im)
    elif method=='li':
        th_val = filters.threshold_li(im)
    elif method=='mean':
        th_val = filters.threshold_mean(im)
    elif method=='triangle':
        th_val = filters.threshold_triangle(im)
    elif method=='isodata':
        th_val = filters.threshold_isodata(im)
    elif method=='quantile':
        th_val = np.quantile(im, quantile)
    else:
        raise ValueError("Unknown thresholding method")


    # th_val = 0.5 * (filters.threshold_otsu(im)+filters.threshold_yen(im))

    # crack binary mask
    if black_cracks:
        crack_bin = im <= th_val # boolean mask
    else:
        crack_bin = im > th_val # boolean mask

    # Remove tiny specks first to prevent them from connecting cracks during closing
    crack_bin = morphology.remove_small_objects(crack_bin, max_size=10)

    # Morphological cleanup - close cracks
    crack_clean = morphology.closing(crack_bin, kernel)
    # for _ in range(4):
    #     crack_clean = morphology.closing(crack_clean, kernel)
    # crack_clean = morphology.opening(crack_clean, morphology.disk(1))

    skeleton = morphology.skeletonize(crack_clean)
    # skeleton = crack_clean
    skeleton = morphology.closing(skeleton, kernel)

    Axx, Axy, Ayy = structure_tensor(skeleton, sigma=2)
    l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])
    coherence = (l1 - l2) / (l1 + l2 + 1e-10)
    skeleton = skeleton & (coherence > min_coherence) # keep only pixels with strong directional coherence

    # # Quantify connected cracks
    # _, num_cracks = measure.label(skeleton, return_num=True)
    # total_crack_area_px = np.sum(skeleton)

    # crack_density = np.mean(skeleton)

    return crack_clean, skeleton, th_val


def find_cracks(image):
    img = enhance_cracks(image)
    cracks, _, _ = crack_detector(img, black_cracks=False)
    return cracks


if __name__ == "__main__":
    # find and visualize cracks in the JUDITH dataset for all appropriate images, and save the results to disk

    import matplotlib.pyplot as plt
    from pathlib import Path
    from JUDITHDataset import JUDITHDataset

    output_path = 'cracks'
    Path(output_path).mkdir(exist_ok=True)

    num_cols = 3 #len(dataset.unique_materials)
    num_rows = 2 #len(dataset.unique_loadtypes)

    scale = 1.e-3

    dataset = JUDITHDataset(preload=True, remove_nan=True)
    filtered_dataset = dataset.filter_by_metadata(scale=scale, detector='QBSD')

    for i in range(len(filtered_dataset)):
        materiali  = filtered_dataset.metadata['material'][i]
        loadtypei  = filtered_dataset.metadata['loadtype'][i]
        base_tempi = filtered_dataset.metadata['base_temp'][i]
        fluxi = filtered_dataset.metadata['flux'][i]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols,3*num_rows))
        axes = axes.ravel()
        fig.tight_layout(h_pad=0., w_pad=0.)

        # image
        dat = filtered_dataset.data[i]

        # enhance and detect cracks
        img = enhance_cracks(dat)
        cracks, skeleton, th_val = crack_detector(img, black_cracks=False)

        axes[0].imshow(dat, cmap='gray')
        axes[0].set_title('original')

        axes[1].imshow(img, cmap='gray')
        axes[1].set_title('enhanced')

        h = im_hist(img)
        axes[2].bar(h[1],h[0],width=2*h[1].min())
        axes[2].vlines(th_val, 0, h[0].max(), color='red', linestyles='dashed')
        axes[2].set_title('intensity histogram, threshold = {:.2f}'.format(th_val))

        # skeleton overlay on original image
        axes[3].imshow(dat, cmap='gray')
        axes[3].imshow(np.ma.masked_where(~cracks.astype(bool), cracks), cmap='autumn', alpha=0.5)
        axes[3].set_title('detection overlay')

        axes[4].imshow(cracks, cmap='gray')
        axes[4].set_title('cracks = enhanced>threshold')

        axes[5].imshow(skeleton, cmap='gray')
        axes[5].set_title('skeleton')

        for ax in axes:
            # Hide x-axis ticks and labels
            ax.set_xticks([]) #
            ax.set_xticklabels([])
            # Hide y-axis ticks and labels
            ax.set_yticks([]) #
            ax.set_yticklabels([])
            # Hide spines
            for spine in ax.spines.values():
                spine.set_visible(False)
        # plt.suptitle(f"{filtered_dataset.metadata['base_temp']}", fontsize=16, x=0.5, y=1.08)
        temp_str = f"{int(base_tempi)}" if not np.isnan(base_tempi) else "nan"
        flux_str = f"{int(fluxi)}" if not np.isnan(fluxi) else "nan"
        plt.savefig(f"{output_path}/{i}_{materiali}_{loadtypei}_{temp_str}_{flux_str}.png", dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close('all')
