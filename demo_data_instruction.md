# Demo Data Instruction

This repository contains the core MATLAB implementation of the PHYSIQ Noise2Noise (N2N) denoising workflow.

## Expected demo-data folder structure

To run the main script, prepare a folder containing two paired 4D TIFF stacks:

```text
example_dataset/
├── N2NA_aligned.tif
└── N2NB_aligned.tif
```

The two TIFF files should correspond to two matched noisy observations of the same volumetric SRS movie, for example two physics-paired I/Q acquisition channels.

## TIFF stack order

Each 4D TIFF file should be saved as sequential 2D z-slices:

```text
T1_Z1, T1_Z2, ..., T1_Zn,
T2_Z1, T2_Z2, ..., T2_Zn,
...
```

For example, if each volume contains 8 z-slices and the movie contains 75 time points, each TIFF file should contain:

```text
8 × 75 = 600 frames
```

The number of z-slices should be specified in the MATLAB script:

```matlab
numZ = 8;
```

## How to run with demo data

1. Place the two paired TIFF files in a folder named `example_dataset`.
2. Put `example_dataset` in the same directory as the MATLAB script.
3. In the main script, set:

```matlab
inputDirs = {
    fullfile(pwd, 'example_dataset')
};
```

4. Run the main MATLAB script:

```matlab
main_Cross_N2N_24GB_clean_english
```

## Output folders

After processing, the script will automatically create output folders containing:

```text
denoised_A_only/
denoised_B_only/
ensemble denoised TIFF volumes
trained models
```

## Notes

A small example dataset can be added to this folder if data sharing is permitted. If the raw microscopy data are too large for GitHub, they can be deposited in a separate data repository such as Zenodo, Figshare, or an institutional repository, and linked from the main README.
