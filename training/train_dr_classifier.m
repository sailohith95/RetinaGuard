%% train_dr_classifier.m
%
%   Placeholder training script for the DR severity classifier.
%
%   This script will train a transfer-learned CNN on the APTOS 2019
%   dataset to classify retinal images into DR grades 0-4.
%
%   PREREQUISITES:
%   1. APTOS 2019 dataset in data/aptos/ (see DATASET_SETUP.md)
%   2. MATLAB Deep Learning Toolbox
%   3. A pretrained base network (GoogLeNet, ResNet-50, or EfficientNet)
%
%   OUTPUT:
%   models/dr/dr_classifier.mat  containing variables: net, classes

fprintf('RetinaGuard — DR Classifier Training Script\n');
fprintf('============================================\n\n');

cfg = RGConfig();

% ---- Step 1: Check dataset -----------------------------------------
aptosDir  = fullfile(cfg.paths.aptos, 'train_images');
aptosCSV  = fullfile(cfg.paths.aptos, 'train.csv');

if ~isfolder(aptosDir) || ~isfile(aptosCSV)
    error(['APTOS 2019 dataset not found.\n' ...
           'Expected:\n  %s\n  %s\n' ...
           'See DATASET_SETUP.md for instructions.'], aptosDir, aptosCSV);
end

fprintf('Dataset found: %s\n', cfg.paths.aptos);

% ---- Step 2: Load labels from CSV ----------------------------------
fprintf('Loading labels from train.csv...\n');
T = readtable(aptosCSV);
% Expected columns: id_code, diagnosis (0-4)
imageIDs = T.id_code;
labels   = T.diagnosis;

classes  = categorical(0:4);
fprintf('Total images: %d\n', height(T));
for g = 0:4
    n = sum(labels == g);
    fprintf('  Grade %d: %d images\n', g, n);
end

% ---- Step 3: Split train/val (80/20) -------------------------------
rng(42);
idx     = randperm(height(T));
nTrain  = round(0.8 * height(T));
idxTrain = idx(1:nTrain);
idxVal   = idx(nTrain+1:end);
fprintf('\nTrain: %d  |  Val: %d\n', numel(idxTrain), numel(idxVal));

% ---- Step 4: Create imageDatastore ---------------------------------
fprintf('Creating imageDatastore...\n');
imgFiles   = fullfile(aptosDir, strcat(imageIDs, '.png'));
% Fallback to .jpg if .png not found
for k = 1:numel(imgFiles)
    if ~isfile(imgFiles{k})
        imgFiles{k} = strrep(imgFiles{k}, '.png', '.jpg');
    end
end

% Filter to existing files
exists = cellfun(@isfile, imgFiles);
if sum(exists) < numel(imgFiles)
    fprintf('Warning: %d image files not found — skipping.\n', sum(~exists));
    imgFiles  = imgFiles(exists);
    labels    = labels(exists);
end

% Build datastores
imdsTrain = imageDatastore(imgFiles(idxTrain), ...
    'Labels', categorical(labels(idxTrain)));
imdsVal   = imageDatastore(imgFiles(idxVal), ...
    'Labels', categorical(labels(idxVal)));

% ---- Step 5: Preprocessing -----------------------------------------
targetSize = cfg.preprocess.targetSize;
augTrain = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation',    [-30 30], ...
    'RandXTranslation',[-20 20], ...
    'RandYTranslation',[-20 20]);

augImdsTrain = augmentedImageDatastore([targetSize 3], imdsTrain, 'DataAugmentation', augTrain);
augImdsVal   = augmentedImageDatastore([targetSize 3], imdsVal);

% ---- Step 6: Load pretrained network --------------------------------
fprintf('Loading pretrained GoogLeNet...\n');
% Replace with resnet50, efficientnetb0, etc. if preferred
try
    baseNet = googlenet;
catch
    error('GoogLeNet not available. Install Deep Learning Toolbox Model for GoogLeNet.');
end

% ---- Step 7: Modify for 5-class classification ----------------------
lgraph = layerGraph(baseNet);
numClasses = 5;

% Find and replace the fully-connected and classification layers
% (layer names vary by network — adjust as needed for resnet50)
newFCLayer   = fullyConnectedLayer(numClasses, 'Name', 'fc_dr', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax   = softmaxLayer('Name','softmax_dr');
newOutput    = classificationLayer('Name', 'output_dr', 'Classes', classes);

% For GoogLeNet: replace loss3-classifier, prob, output
lgraph = replaceLayer(lgraph, 'loss3-classifier', newFCLayer);
lgraph = replaceLayer(lgraph, 'prob', newSoftmax);
lgraph = replaceLayer(lgraph, 'output', newOutput);

% ---- Step 8: Training options ---------------------------------------
opts = trainingOptions('adam', ...
    'InitialLearnRate',     1e-4, ...
    'MaxEpochs',            20, ...
    'MiniBatchSize',        32, ...
    'Shuffle',              'every-epoch', ...
    'ValidationData',       augImdsVal, ...
    'ValidationFrequency',  10, ...
    'Plots',                'training-progress', ...
    'Verbose',              true, ...
    'OutputNetwork',        'best-validation-loss');

% ---- Step 9: Train --------------------------------------------------
fprintf('\nStarting training...\n');
fprintf('This may take several hours depending on hardware.\n\n');
net = trainNetwork(augImdsTrain, lgraph, opts);

% ---- Step 10: Save model -------------------------------------------
modelDir = cfg.paths.drModel;
if ~isfolder(fileparts(modelDir))
    mkdir(fileparts(modelDir));
end
save(cfg.paths.drModel, 'net', 'classes');
fprintf('\nModel saved: %s\n', cfg.paths.drModel);
fprintf('Restart RetinaGuard to load the trained model.\n');
