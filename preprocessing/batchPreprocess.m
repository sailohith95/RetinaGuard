%% batchPreprocess.m
%
%   Batch preprocessing utility for training data preparation.
%
%   Reads all images from a source directory, applies the RetinaGuard
%   preprocessing pipeline, and saves processed images to an output directory.
%
%   Usage:
%     batchPreprocess('data/aptos/train_images', 'data/aptos/processed', cfg)

function batchPreprocess(srcDir, outDir, cfg)
if nargin < 3, cfg = RGConfig(); end

if ~isfolder(outDir), mkdir(outDir); end

exts = {'*.jpg','*.jpeg','*.png','*.tif','*.tiff'};
files = {};
for i = 1:numel(exts)
    f = dir(fullfile(srcDir, exts{i}));
    for k = 1:numel(f)
        files{end+1} = fullfile(f(k).folder, f(k).name);
    end
end

fprintf('Preprocessing %d images from %s\n', numel(files), srcDir);
fprintf('Output: %s\n\n', outDir);

for i = 1:numel(files)
    [~, name, ext] = fileparts(files{i});
    outPath = fullfile(outDir, [name '.png']);

    if isfile(outPath)
        continue;  % Skip already processed
    end

    try
        [out, ~] = preprocessRetinalImage(files{i}, cfg);
        imwrite(out, outPath);
        if mod(i,50)==0
            fprintf('  Processed %d/%d\n', i, numel(files));
        end
    catch ME
        fprintf('  [SKIP] %s: %s\n', name, ME.message);
    end
end

fprintf('\nBatch preprocessing complete.\n');
end
