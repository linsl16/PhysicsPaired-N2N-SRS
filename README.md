# Physics-Paired N2N-SRS

This repository provides a MATLAB implementation of a physics-paired Noise2Noise denoising workflow for volumetric stimulated Raman scattering (SRS) microscopy.

The denoising code is based on the original Noise2Noise principle and has been adapted for PHYSIQ-SRS data, where two matched noisy observations of the same Raman signal are acquired through paired I/Q channels. These physics-paired measurements serve as input-target pairs for self-supervised 3D denoising without requiring clean ground-truth images.
## Core functionality

The main script performs the following steps:

1. Load paired 4D TIFF stacks from two matched noisy channels, denoted as channel A and channel B.
2. Apply robust percentile-based intensity normalization.
3. Extract paired 3D training patches from the two channels.
4. Train two anisotropic 3D U-Net denoising models:
   - A-to-B model
   - B-to-A model
5. Apply sliding-window 3D inference to each time point.
6. Save denoised channel-specific outputs and the ensemble-averaged denoised volume.

The implementation is designed for volumetric SRS datasets with a relatively small number of axial slices, for example 3D volumes with 8 z-planes. The network therefore uses anisotropic pooling, where downsampling is applied in the lateral dimensions while preserving the axial dimension.

## Repository contents
Requirements

This code was tested in MATLAB with the Deep Learning Toolbox and Image Processing Toolbox.
## Scope of the released code

This repository contains two main components:

1. **PHYSIQ-N2N denoising**
   - paired 4D TIFF loading
   - robust intensity normalization
   - paired 3D patch extraction
   - anisotropic 3D U-Net construction
   - bidirectional Noise2Noise training
   - sliding-window volumetric inference
   - ensemble denoised-output generation

2. **MSD-based trajectory analysis**
   - TrackMate XML import
   - trajectory filtering
   - displacement and velocity calculation
   - MSD calculation
   - short-lag α fitting
   - motion-state classification
   - publication-style plotting

LD localization and initial trajectory linking were performed in Fiji using the TrackMate plugin. TrackMate settings are documented below and in the Methods of the manuscript.
## Recommended hardware:

NVIDIA GPU with sufficient memory
24 GB GPU memory was used for the default configuration
Smaller GPUs may require reducing patchSize or miniBatchSize
## License

This repository is released for academic and research use. Please see the LICENSE file for details.
main_Crossvalidation_N2N.m
