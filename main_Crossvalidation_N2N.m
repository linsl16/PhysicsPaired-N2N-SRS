%% PHYSIQ 3D Noise2Noise Denoising Pipeline
% =========================================================================
% Core functionality:
%   This script trains and applies a paired 3D Noise2Noise (N2N) denoising
%   workflow for PHYSIQ/SRS volumetric image sequences. It uses two matched
%   noisy input stacks, channel A and channel B, as physics-paired noisy
%   observations of the same underlying signal. The script trains two
%   anisotropic 3D U-Net models in opposite directions, A -> B and B -> A,
%   applies sliding-window inference to each time point, and saves the
%   denoised A-only, B-only, and ensemble-fused 3D TIFF volumes.
%
% Version:
%   N2N v4.0 - Anisotropic 3D U-Net optimized for shallow Z stacks.
%
% Main features:
%   1. Anisotropic pooling: down-samples only in XY while preserving the Z
%      dimension using a pooling size of [2 2 1].
%   2. Robust normalization: uses the 99.99th percentile instead of the
%      absolute maximum to reduce the influence of hot pixels.
%   3. Data augmentation: expands training patches by rotation and flipping.
%   4. Batch processing: processes multiple input folders sequentially.
%   5. Dual-model ensemble: combines A -> B and B -> A denoised outputs.
%
% Expected input files in each input folder:
%   N2NA_aligned.tif
%   N2NB_aligned.tif
%
% Expected TIFF organization:
%   A 4D volume sequence saved as a multi-page TIFF, ordered as
%   T1-Z1, T1-Z2, ..., T1-Zn, T2-Z1, ..., where numZ is defined below.
%
% Notes for open-source release:
%   Replace the example paths in inputDirs with your own dataset folders.
%   This code requires MATLAB Deep Learning Toolbox and Image Processing
%   Toolbox. GPU acceleration is used automatically when available.
% =========================================================================

clear; clc; close all;

%% 1. Batch configuration
% =========================================================================
% --- Input folder list ---
inputDirs = {
    % Replace the path below with one or more folders containing
    % N2NA_aligned.tif and N2NB_aligned.tif.
    % Example:
    % 'D:\your_dataset_folder\cell_001'
      'example_dataset'
};

numZ = 8; % Number of Z-slices per 3D volume.

% --- Core training parameters ---
patchSize = [64, 64, 8];    % Patch size [height, width, z].
numPatches = 3000;          % Number of randomly sampled N2N patch pairs before augmentation.
validationSplit = 0.2;
maxEpochs = 50;
miniBatchSize = 16;
initialLearnRate = 1e-4;
learnRateDropFactor = 0.5;
learnRateDropPeriod = 15;
validationPatience = 10;
% =========================================================================


%% 2. Process each folder sequentially
for k = 1:numel(inputDirs)
    inputDir = strtrim(inputDirs{k});
    fprintf('\n====================================================================\n');
    fprintf('Processing folder %d / %d: %s\n', k, numel(inputDirs), inputDir);
    fprintf('====================================================================\n');

    % Generate output and model folders.
    outputDir = fullfile(inputDir, sprintf('denoised_N2N_v4_E%d_P%dx%dx%d', maxEpochs, patchSize(1), patchSize(2), patchSize(3)));
    modelDir  = fullfile(inputDir, 'trained_model');

    if ~isfolder(inputDir)
        warning('Input folder does not exist. Skipping: %s', inputDir);
        continue;
    end

    inputFileName_A = 'N2NA_aligned.tif';
    inputFileName_B = 'N2NB_aligned.tif';

    % Create output folders.
    if ~exist(outputDir, 'dir'), mkdir(outputDir); end
    if ~exist(modelDir, 'dir'), mkdir(modelDir); end
    outputDir_A = fullfile(outputDir, 'denoised_A_only');
    outputDir_B = fullfile(outputDir, 'denoised_B_only');
    if ~exist(outputDir_A, 'dir'), mkdir(outputDir_A); end
    if ~exist(outputDir_B, 'dir'), mkdir(outputDir_B); end

    modelPath_A_to_B = fullfile(modelDir, '3DN2N_v4_model_A_to_B.mat');
    modelPath_B_to_A = fullfile(modelDir, '3DN2N_v4_model_B_to_A.mat');

    try
        %% 3. Load and preprocess data
        fprintf('--- Data loading and preprocessing ---\n');
        fullInputPath_A = fullfile(inputDir, inputFileName_A);
        fullInputPath_B = fullfile(inputDir, inputFileName_B);

        if ~exist(fullInputPath_A, 'file') || ~exist(fullInputPath_B, 'file')
            error('Input file A or B was not found.');
        end

        [rawData_A, dims_A] = read_4d_tif(fullInputPath_A, numZ);
        [rawData_B, dims_B] = read_4d_tif(fullInputPath_B, numZ);

        if ~isequal(dims_A, dims_B)
            error('Input A and target B have mismatched dimensions.');
        end
        fprintf('Data dimensions: H=%d, W=%d, Z=%d, T=%d\n', dims_A(1), dims_A(2), dims_A(3), dims_A(4));

        % Robust 99.99th-percentile normalization.
        fprintf('Running robust percentile normalization...\n');
        sortedData = sort(rawData_A(:));
        maxValue = sortedData(round(length(sortedData) * 0.9999));
        if maxValue == 0, maxValue = 1; end

        rawData_A = min(rawData_A / single(maxValue), 1.0); % Clip high-intensity outliers.
        rawData_B = min(rawData_B / single(maxValue), 1.0);
        fprintf('Data loading and normalization completed. Normalization factor maxValue = %g\n\n', maxValue);

        %% 4. Create N2N training and validation patch pairs
        fprintf('--- Creating N2N training and validation pairs ---\n');
        [trainPatches_A, trainPatches_B, valPatches_A, valPatches_B] = ...
            getN2NPairs(rawData_A, rawData_B, patchSize, numPatches, validationSplit);

        % Four-fold data augmentation.
        [trainPatches_A, trainPatches_B] = augmentPatches(trainPatches_A, trainPatches_B);
        fprintf('Data augmentation completed: training patches expanded four-fold.\n');
        fprintf('Final training set: %d pairs; validation set: %d pairs.\n\n', size(trainPatches_A,5), size(valPatches_A,5));

        %% 5. Train or load two independent N2N networks
        networkInputSize = size(trainPatches_A, [1,2,3,4]);

        options = trainingOptions('adam', ...
            'InitialLearnRate', initialLearnRate, ...
            'MaxEpochs', maxEpochs, ...
            'MiniBatchSize', miniBatchSize, ...
            'Shuffle', 'every-epoch', ...
            'Plots', 'none', ...
            'Verbose', true, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', learnRateDropFactor, ...
            'LearnRateDropPeriod', learnRateDropPeriod, ...
            'ExecutionEnvironment', 'auto', ...
            'ValidationPatience', validationPatience);

        % Task 1: train or load the A -> B model.
        fprintf('--- Task 1: Train model A -> B ---\n');
        if exist(modelPath_A_to_B, 'file')
            fprintf('Loading trained model: %s\n', modelPath_A_to_B);
            load(modelPath_A_to_B, 'net');
            net_A_to_B = net;
        else
            fprintf('Creating anisotropic 3D U-Net model (A -> B)...\n');
            lgraph_A_to_B = createAnisotropic3DN2NModel(networkInputSize);
            options.ValidationData = {valPatches_A, valPatches_B};
            fprintf('Training network (A -> B)...\n');
            [net_A_to_B, ~] = trainNetwork(trainPatches_A, trainPatches_B, lgraph_A_to_B, options);
            net = net_A_to_B;
            save(modelPath_A_to_B, 'net');
        end

        % Task 2: train or load the B -> A model.
        fprintf('\n--- Task 2: Train model B -> A ---\n');
        if exist(modelPath_B_to_A, 'file')
            fprintf('Loading trained model: %s\n', modelPath_B_to_A);
            load(modelPath_B_to_A, 'net');
            net_B_to_A = net;
        else
            fprintf('Creating anisotropic 3D U-Net model (B -> A)...\n');
            lgraph_B_to_A = createAnisotropic3DN2NModel(networkInputSize);
            options.ValidationData = {valPatches_B, valPatches_A};
            fprintf('Training network (B -> A)...\n');
            [net_B_to_A, ~] = trainNetwork(trainPatches_B, trainPatches_A, lgraph_B_to_A, options);
            net = net_B_to_A;
            save(modelPath_B_to_A, 'net');
        end
        fprintf('\nAll models are ready.\n\n');

        %% 6. Denoise and ensemble all time points using the two models
        fprintf('--- Starting bidirectional denoising and ensemble inference for all time points ---\n');
        predPatchSize = networkInputSize(1:3);
        % Alternative 50%% overlap setting:
        % stride = max(1, floor(predPatchSize / 2));
        stride = [32, 32, 8]; % 50%% overlap in XY and full-stack prediction in Z.
        fprintf('Prediction parameters: PatchSize=[%d %d %d], Stride=[%d %d %d]\n', predPatchSize, stride);

        [~, inputName_A, ~] = fileparts(inputFileName_A);
        [~, inputName_B, ~] = fileparts(inputFileName_B);
        numT = dims_A(4);

        for t = 1:numT
            fprintf('  Processing time point %d / %d...\n', t, numT);

            vol_A_noisy = rawData_A(:,:,:,t);
            vol_B_noisy = rawData_B(:,:,:,t);

            % Prediction 1: denoise channel A using the A -> B model.
            denoised_A_normalized = sliding_window_predict(vol_A_noisy, net_A_to_B, predPatchSize, stride);
            denoised_A_final = denoised_A_normalized * maxValue;
            outputFileName_A = sprintf('%s_denoised_AtoB_T%04d.tif', inputName_A, t);
            save_3d_tif(uint16(denoised_A_final), fullfile(outputDir_A, outputFileName_A));

            % Prediction 2: denoise channel B using the B -> A model.
            denoised_B_normalized = sliding_window_predict(vol_B_noisy, net_B_to_A, predPatchSize, stride);
            denoised_B_final = denoised_B_normalized * maxValue;
            outputFileName_B = sprintf('%s_denoised_BtoA_T%04d.tif', inputName_B, t);
            save_3d_tif(uint16(denoised_B_final), fullfile(outputDir_B, outputFileName_B));

            % Ensemble the two denoised predictions and save the fused result.
            final_vol_normalized = (denoised_A_normalized + denoised_B_normalized) / 2;
            final_vol = final_vol_normalized * maxValue;
            outputFileName_fused = sprintf('%s_denoised_Ensemble_T%04d.tif', inputName_A, t);
            fprintf('    -> Saving ensemble result: %s\n', outputFileName_fused);
            save_3d_tif(uint16(final_vol), fullfile(outputDir, outputFileName_fused));
        end

        fprintf('Folder processing completed.\n');

    catch ME
        fprintf(2, 'Error while processing folder %s: %s\n', inputDir, ME.message);
    end
end

%% ========================================================================
%                         Helper functions
% =========================================================================

% ------------------------------------------------------------------------
% Data augmentation: original, 180-degree rotation, left-right flip, and
% up-down flip. This expands the training set by a factor of four.
% ------------------------------------------------------------------------
function [augPatches_A, augPatches_B] = augmentPatches(patches_A, patches_B)
    [H, W, Z, C, N] = size(patches_A);

    augPatches_A = zeros(H, W, Z, C, N * 4, 'single');
    augPatches_B = zeros(H, W, Z, C, N * 4, 'single');

    % 1. Original patches.
    augPatches_A(:,:,:,:, 1:N) = patches_A;
    augPatches_B(:,:,:,:, 1:N) = patches_B;

    for i = 1:N
        % 2. 180-degree rotation.
        augPatches_A(:,:,:,:, N + i) = rot90(patches_A(:,:,:,:,i), 2);
        augPatches_B(:,:,:,:, N + i) = rot90(patches_B(:,:,:,:,i), 2);

        % 3. Left-right flip.
        augPatches_A(:,:,:,:, 2*N + i) = fliplr(patches_A(:,:,:,:,i));
        augPatches_B(:,:,:,:, 2*N + i) = fliplr(patches_B(:,:,:,:,i));

        % 4. Up-down flip.
        augPatches_A(:,:,:,:, 3*N + i) = flipud(patches_A(:,:,:,:,i));
        augPatches_B(:,:,:,:, 3*N + i) = flipud(patches_B(:,:,:,:,i));
    end
end

% ------------------------------------------------------------------------
% Anisotropic 3D U-Net model for shallow Z stacks.
% The pooling and transposed-convolution size is [2, 2, 1], so the network
% downsamples in XY while preserving the number of Z planes.
% ------------------------------------------------------------------------
function lgraph = createAnisotropic3DN2NModel(inputSize)
    poolSize = [2, 2, 1];

    layers = [
        image3dInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none');
        convolution3dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_1'); reluLayer('Name', 'relu1_1');
        convolution3dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1_2'); reluLayer('Name', 'relu1_2');
        maxPooling3dLayer(poolSize, 'Stride', poolSize, 'Name', 'pool1');

        convolution3dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_1'); reluLayer('Name', 'relu2_1');
        convolution3dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2_2'); reluLayer('Name', 'relu2_2');
        maxPooling3dLayer(poolSize, 'Stride', poolSize, 'Name', 'pool2');

        convolution3dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_1'); reluLayer('Name', 'relu3_1');
        convolution3dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3_2'); reluLayer('Name', 'relu3_2');
        maxPooling3dLayer(poolSize, 'Stride', poolSize, 'Name', 'pool3');

        convolution3dLayer(3, 256, 'Padding', 'same', 'Name', 'conv_b1'); reluLayer('Name', 'relu_b1');
        convolution3dLayer(3, 256, 'Padding', 'same', 'Name', 'conv_b2'); reluLayer('Name', 'relu_b2');
    ];
    lgraph = layerGraph(layers);

    % Decoder level 3.
    layers = [
        transposedConv3dLayer(poolSize, 128, 'Stride', poolSize, 'Name', 'transp_conv3');
        reluLayer('Name','relu_transp3')
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'relu_b2', 'transp_conv3');

    layers = [
        concatenationLayer(4, 2, 'Name', 'concat3');
        convolution3dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4_1'); reluLayer('Name', 'relu4_1');
        convolution3dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4_2'); reluLayer('Name', 'relu4_2');
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'relu_transp3', 'concat3/in1');
    lgraph = connectLayers(lgraph, 'relu3_2', 'concat3/in2');

    % Decoder level 2.
    layers = [
        transposedConv3dLayer(poolSize, 64, 'Stride', poolSize, 'Name', 'transp_conv2');
        reluLayer('Name','relu_transp2')
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'relu4_2', 'transp_conv2');

    layers = [
        concatenationLayer(4, 2, 'Name', 'concat2');
        convolution3dLayer(3, 64, 'Padding', 'same', 'Name', 'conv5_1'); reluLayer('Name', 'relu5_1');
        convolution3dLayer(3, 64, 'Padding', 'same', 'Name', 'conv5_2'); reluLayer('Name', 'relu5_2');
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'relu_transp2', 'concat2/in1');
    lgraph = connectLayers(lgraph, 'relu2_2', 'concat2/in2');

    % Decoder level 1.
    layers = [
        transposedConv3dLayer(poolSize, 32, 'Stride', poolSize, 'Name', 'transp_conv1');
        reluLayer('Name','relu_transp1')
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'relu5_2', 'transp_conv1');

    layers = [
        concatenationLayer(4, 2, 'Name', 'concat1');
        convolution3dLayer(3, 32, 'Padding', 'same', 'Name', 'conv6_1'); reluLayer('Name', 'relu6_1');
        convolution3dLayer(3, 32, 'Padding', 'same', 'Name', 'conv6_2'); reluLayer('Name', 'relu6_2');
        convolution3dLayer(1, 1, 'Name', 'final_conv');
        regressionLayer('Name', 'output');
    ];
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, 'relu_transp1', 'concat1/in1');
    lgraph = connectLayers(lgraph, 'relu1_2', 'concat1/in2');
end

% ------------------------------------------------------------------------
% TIFF input/output and inference functions.
% ------------------------------------------------------------------------

function [data4D, dims] = read_4d_tif(fullPath, numZ)
    info = imfinfo(fullPath);
    numFrames = numel(info);
    H = info(1).Height;
    W = info(1).Width;
    numT = numFrames / numZ;

    if rem(numT, 1) ~= 0
        error('The number of Z planes (numZ=%d) is incompatible with the total number of TIFF frames (%d).', numZ, numFrames);
    end

    dims = [H, W, numZ, numT];
    data4D = zeros(H, W, numZ, numT, 'single');

    TiffReader = Tiff(fullPath, 'r');
    for t = 1:numT
        for z = 1:numZ
            frameIndex = (t - 1) * numZ + z;
            TiffReader.setDirectory(frameIndex);
            data4D(:,:,z,t) = TiffReader.read();
        end
    end
    TiffReader.close();
end

function save_3d_tif(vol3D, fullPath)
    for z = 1:size(vol3D, 3)
        if z == 1
            imwrite(vol3D(:,:,z), fullPath, 'Tiff', 'Compression', 'none');
        else
            imwrite(vol3D(:,:,z), fullPath, 'Tiff', 'WriteMode', 'append', 'Compression', 'none');
        end
    end
end

function [trainPatches_A, trainPatches_B, valPatches_A, valPatches_B] = getN2NPairs(data_A, data_B, patchSize, numPatches, valSplit)
    [H, W, Z, ~] = size(data_A);
    pH = patchSize(1);
    pW = patchSize(2);
    pZ = patchSize(3);

    padSize = max(0, [pH, pW, pZ] - [H, W, Z]);
    if any(padSize > 0)
        pad_dims_xy = [padSize(1:2), 0, 0];
        pad_dims_z = [0, 0, padSize(3), 0];
        data_A = padarray(padarray(data_A, pad_dims_xy, 'replicate', 'post'), pad_dims_z, 'replicate', 'post');
        data_B = padarray(padarray(data_B, pad_dims_xy, 'replicate', 'post'), pad_dims_z, 'replicate', 'post');
    end

    numVal = floor(numPatches * valSplit);
    numTrain = numPatches - numVal;

    trainPatches_A = zeros(pH, pW, pZ, 1, numTrain, 'single');
    trainPatches_B = zeros(pH, pW, pZ, 1, numTrain, 'single');
    valPatches_A = zeros(pH, pW, pZ, 1, numVal, 'single');
    valPatches_B = zeros(pH, pW, pZ, 1, numVal, 'single');

    for i = 1:numTrain
        [patchA, patchB] = extract_random_pair(data_A, data_B, patchSize);
        trainPatches_A(:,:,:,1,i) = patchA;
        trainPatches_B(:,:,:,1,i) = patchB;
    end

    for i = 1:numVal
        [patchA, patchB] = extract_random_pair(data_A, data_B, patchSize);
        valPatches_A(:,:,:,1,i) = patchA;
        valPatches_B(:,:,:,1,i) = patchB;
    end
end

function [patchA, patchB] = extract_random_pair(data_A, data_B, patchSize)
    [H, W, Z, T] = size(data_A);
    pH = patchSize(1);
    pW = patchSize(2);
    pZ = patchSize(3);

    randT = randi(T);
    randH = randi(H - pH + 1);
    randW = randi(W - pW + 1);
    randZ = randi(Z - pZ + 1);

    patchA = data_A(randH:randH+pH-1, randW:randW+pW-1, randZ:randZ+pZ-1, randT);
    patchB = data_B(randH:randH+pH-1, randW:randW+pW-1, randZ:randZ+pZ-1, randT);
end

function denoised_vol = sliding_window_predict(vol_noisy, net, predPatchSize, stride)
    [H_orig, W_orig, Z_orig] = size(vol_noisy);
    pH = predPatchSize(1);
    pW = predPatchSize(2);
    pZ = predPatchSize(3);
    sH = stride(1);
    sW = stride(2);
    sZ = stride(3);

    numPatchesH = ceil(max(0, H_orig - pH) / sH) + 1;
    numPatchesW = ceil(max(0, W_orig - pW) / sW) + 1;
    numPatchesZ = ceil(max(0, Z_orig - pZ) / sZ) + 1;

    paddedH = (numPatchesH - 1) * sH + pH;
    paddedW = (numPatchesW - 1) * sW + pW;
    paddedZ = (numPatchesZ - 1) * sZ + pZ;
    padSize = [paddedH - H_orig, paddedW - W_orig, paddedZ - Z_orig];

    padded_vol = padarray(vol_noisy, padSize, 'replicate', 'post');
    denoised_padded_vol = zeros(size(padded_vol), 'single');
    weight_map = zeros(size(padded_vol), 'single');

    if gpuDeviceCount > 0
        denoised_padded_vol = gpuArray(denoised_padded_vol);
        weight_map = gpuArray(weight_map);
        padded_vol = gpuArray(padded_vol);
    end

    win = hanning(pH) * hanning(pW)' .* reshape(hanning(pZ), 1, 1, pZ);

    for z_start = 1:sZ:(paddedZ - pZ + 1)
        for w_start = 1:sW:(paddedW - pW + 1)
            for h_start = 1:sH:(paddedH - pH + 1)
                h_end = h_start + pH - 1;
                w_end = w_start + pW - 1;
                z_end = z_start + pZ - 1;

                patch = padded_vol(h_start:h_end, w_start:w_end, z_start:z_end);
                denoised_patch_4d = predict(net, reshape(patch, [pH, pW, pZ, 1]));
                denoised_patch = squeeze(denoised_patch_4d);

                denoised_padded_vol(h_start:h_end, w_start:w_end, z_start:z_end) = ...
                    denoised_padded_vol(h_start:h_end, w_start:w_end, z_start:z_end) + denoised_patch .* win;
                weight_map(h_start:h_end, w_start:w_end, z_start:z_end) = ...
                    weight_map(h_start:h_end, w_start:w_end, z_start:z_end) + win;
            end
        end
    end

    weight_map(weight_map == 0) = 1;
    denoised_padded_vol = denoised_padded_vol ./ weight_map;
    denoised_vol = gather(denoised_padded_vol(1:H_orig, 1:W_orig, 1:Z_orig));
end
