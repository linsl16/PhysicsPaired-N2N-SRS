# PhysicsPaired-N2N-SRS
Self-supervised 3D Noise2Noise denoising for physics-paired volumetric SRS microscopy using paired I/Q acquisitions.

This repository provides a MATLAB implementation of physics-paired Noise2Noise (N2N) denoising for volumetric stimulated Raman scattering (SRS) microscopy.

The code was developed for paired SRS image volumes acquired by the PHYSIQ framework, in which two matched noisy observations of the same underlying Raman signal are obtained through orthogonal I/Q acquisition channels. These paired noisy volumes enable fully self-supervised Noise2Noise training without requiring a clean ground-truth image.

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

main_Cross_N2N_24GB_clean_english.m
