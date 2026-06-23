# Electron Beam Data Description

This dataset contains microscopy images and experiment-level metadata from tungsten thermal-shock experiments based on the test campaign described by Wirtz (2013). The experiments investigate how different tungsten grades respond to fusion-relevant transient heat loads, especially edge-localized-mode-like (ELM-like) thermal shock conditions.

The experiments were performed in the JUDITH electron-beam facility at Forschungszentrum Jülich (FZJ), Germany. JUDITH is the *Jülicher Divertor Test Facility in Hot Cells* and is used to simulate high heat flux and transient thermal loading conditions relevant to plasma-facing materials in fusion devices such as ITER and DEMO.

In the thesis, industrially manufactured tungsten grades such as W-UHP, pure W, WVMW, WTa1, and WTa5 were exposed to cyclic electron-beam thermal shocks at different base temperatures and absorbed power densities. The goal was to map damage behaviour, including surface modification, roughening, crack formation, and crack networks, and to relate these effects to material grade, grain structure, recrystallisation state, base temperature, and thermal flux.

### Reference

Wirtz, O. M. (2013). *Thermal Shock Behaviour of Different Tungsten Grades under Varying Conditions*. Schriften des Forschungszentrums Jülich, Energy & Environment, Vol. 161. Forschungszentrum Jülich.

## Original data format

Original data repository contains microscopy image data from tungsten thermal-shock experiments. Image-level metadata is inferred from the directory structure, the image filename, and the accompanying test matrix.

### Test matrix

The file `Testmatrix JUDITH 1.xlsx` contains the test matrix `(material, loadtype, base temperature, thermal flux)` of the experiment campaign.
Each experiment is assigned a unique label that is used in the naming of images.

### Expected directory structure

```text
<data_root>/
├── Testmatrix JUDITH 1.xlsx
├── <material>/
│   ├── <loadtype>/
│   │   ├── <label>_<id>.tif
│   │   ├── <label>_<id>.tif
│   │   └── ...
│   └── ...
└── ...
```

Where:

* `<material>` is the material or tungsten grade, for example `W-UHP`, `pure W`, `WVMW`, `WTa1`, or `WTa5`.
* `<loadtype>` is the sample cut orientation.
* `<label>` is the experiment label used to look up base temperature and thermal flux in the Excel test matrix.
* `<id>` is an image identifier. It is used to keep filenames unique.

Example:

```text
data/
├── Testmatrix JUDITH 1.xlsx
├── M192 - W-UHP/
│   ├── recrystalised/
│   │   ├── AR2_01.tif
│   │   └── AR3_01.tif
│   └── transversal/
│       └── AT2_03.tif
└── M195 - WTa5/
    └── recrystallised/
        └── DR3_04.tif
```


## Data processing

Data loading and processing is implemented in [/src/data_utils.py](../src/data_utils.py).
Running the script extracts the metadata from dataset and saves it as `/data/JUDITH/metadata.npz`.

Particularly, `def build_label2tempflux()` is used to extract the test matrix from `/data/JUDITH/Testmatrix JUDITH 1.xlsx`:

```text
experiment label -> (base temperature, thermal flux)
```

The label is extracted from the image filename by taking everything before the first underscore. For example:

```text
184A_001.tif -> label = 184A
```

If a label is not found in the test matrix, the corresponding `base_temp` and `flux` metadata values are set to `NaN`.


