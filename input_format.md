# Input Format

This document describes the expected input format for the PHYSIQ Noise2Noise denoising workflow.

## Required input files

Each input directory should contain two paired TIFF stacks:

```text
N2NA_aligned.tif
N2NB_aligned.tif
```

These two files represent two matched noisy measurements of the same underlying 4D volumetric SRS movie.

## Data dimensions

The expected data structure is:

```text
Height × Width × Z × Time
```

The TIFF files themselves are stored as sequential 2D frames. The script reconstructs the 4D array using the user-defined number of z-slices:

```matlab
numZ = 8;
```

The total number of frames in each TIFF file must be divisible by `numZ`.

For example:

```text
Height = 320 pixels
Width  = 320 pixels
Z      = 8 slices
Time   = 75 volumes
```

The TIFF file should therefore contain:

```text
8 × 75 = 600 frames
```

## Frame order

The expected frame order is:

```text
T1_Z1
T1_Z2
...
T1_Z8
T2_Z1
T2_Z2
...
T2_Z8
...
```

In other words, z-slices are stored first within each time point, and time points are stored sequentially.

## Channel pairing

The two input stacks must be spatially and temporally matched:

```text
N2NA_aligned.tif  <->  N2NB_aligned.tif
```

For each time point and z-plane, the corresponding frames in the two files should image the same sample volume.

The two stacks must have identical dimensions:

```text
Height_A = Height_B
Width_A  = Width_B
Z_A      = Z_B
Time_A   = Time_B
```

## Intensity format

The script reads TIFF images and converts them to single-precision arrays internally.

The default implementation uses robust percentile-based normalization:

```matlab
maxValue = percentile(rawData_A, 99.99%);
rawData_A = min(rawData_A / maxValue, 1.0);
rawData_B = min(rawData_B / maxValue, 1.0);
```

This avoids instability caused by rare hot pixels or saturated pixels.

## Default file names

The default file names are:

```matlab
inputFileName_A = 'N2NA_aligned.tif';
inputFileName_B = 'N2NB_aligned.tif';
```

If your files have different names, change these two variables in the main script.

## Recommended test data size

For a lightweight test run, a small cropped dataset can be used, for example:

```text
128 × 128 × 8 × 5
```

For full analysis, use the original volumetric SRS movie.
