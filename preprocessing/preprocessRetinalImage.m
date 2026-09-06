%% preprocessRetinalImage.m
%
%   Standalone preprocessing function that applies the full
%   RetinaGuard preprocessing pipeline and returns step images.
%
%   Useful for batch preprocessing before training.
%
%   [out, steps] = preprocessRetinalImage(imgOrPath, cfg)
%
%   Input:
%     imgOrPath - uint8 RGB image OR a full file path string
%     cfg       - optional RGConfig() struct
%
%   Output:
%     out   - preprocessed uint8 RGB image at targetSize
%     steps - struct with intermediate step images

function [out, steps] = preprocessRetinalImage(imgOrPath, cfg)
if nargin < 2, cfg = RGConfig(); end

% Load from path if string
if ischar(imgOrPath) || isstring(imgOrPath)
    img = imread(char(imgOrPath));
else
    img = imgOrPath;
end

% Ensure uint8 RGB
if ~isa(img,'uint8'), img = im2uint8(img); end
if size(img,3) == 1,  img = cat(3,img,img,img); end
if size(img,3) == 4,  img = img(:,:,1:3); end

[out, steps] = enhanceRetinalImage(img, cfg);
end
