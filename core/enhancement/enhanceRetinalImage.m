function [enhanced, steps] = enhanceRetinalImage(img, cfg)
% enhanceRetinalImage  Full preprocessing pipeline for a retinal fundus image.
%
%   [enhanced, steps] = enhanceRetinalImage(img, cfg)
%
%   Input:
%     img  - uint8 or double RGB image (H×W×3)
%     cfg  - RGConfig() struct
%
%   Output:
%     enhanced - uint8 RGB enhanced image (targetSize × 3)
%     steps    - struct recording each preprocessing step:
%                 .original, .cropped, .resized, .clahe,
%                 .denoised, .normalized
%
%   The original image is NEVER modified; enhanced is a separate output.

if nargin < 2, cfg = RGConfig(); end

steps.original = img;

try
    % Step 1: Ensure RGB uint8
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end

    % Step 2: Remove black border / crop to retinal circle
    cropped = removeBlackBorder(img);
    steps.cropped = cropped;

    % Step 3: Resize to target
    tSize  = cfg.preprocess.targetSize;
    resized = imresize(cropped, tSize);
    steps.resized = resized;

    % Step 4: Convert to double for processing
    imgD = im2double(resized);

    % Step 5: Illumination normalization (subtract local background)
    greenCh = imgD(:,:,2);
    bgGreen = imgaussfilt(greenCh, 30);   % large sigma → illumination map
    for ch = 1:3
        imgD(:,:,ch) = imgD(:,:,ch) - imgaussfilt(imgD(:,:,ch), 30) + 0.5;
    end
    imgD = max(0, min(1, imgD));
    steps.illumNorm = im2uint8(imgD);

    % Step 6: CLAHE on each channel
    clipLimit = cfg.preprocess.claheClipLimit;
    tileSize  = cfg.preprocess.claheTileSize;
    for ch = 1:3
        imgD(:,:,ch) = adapthisteq(imgD(:,:,ch), ...
            'ClipLimit', clipLimit, ...
            'NumTiles', tileSize, ...
            'Distribution', 'rayleigh');
    end
    steps.clahe = im2uint8(imgD);

    % Step 7: Denoising
    noiseMethod = cfg.preprocess.denoiseMethod;
    sigma       = cfg.preprocess.denoiseSigma;
    if strcmp(noiseMethod, 'gaussian')
        for ch = 1:3
            imgD(:,:,ch) = imgaussfilt(imgD(:,:,ch), sigma);
        end
    else
        for ch = 1:3
            imgD(:,:,ch) = medfilt2(imgD(:,:,ch), [3 3]);
        end
    end
    steps.denoised = im2uint8(imgD);

    % Step 8: Final intensity normalization (stretch to [0,1])
    for ch = 1:3
        chData = imgD(:,:,ch);
        chMin  = min(chData(:));
        chMax  = max(chData(:));
        if chMax > chMin
            imgD(:,:,ch) = (chData - chMin) / (chMax - chMin);
        end
    end
    steps.normalized = im2uint8(imgD);

    enhanced = im2uint8(imgD);

catch ME
    warning('enhanceRetinalImage: %s — returning resized original.', ME.message);
    if ~isfield(steps,'cropped'), steps.cropped = img; end
    enhanced = imresize(img, cfg.preprocess.targetSize);
    steps.normalized = enhanced;
end

steps.enhanced = enhanced;
end
