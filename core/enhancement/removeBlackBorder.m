function cropped = removeBlackBorder(img)
% removeBlackBorder  Removes dark border padding around a fundus image.
%
%   cropped = removeBlackBorder(img)
%
%   Input:
%     img  - uint8 RGB image
%
%   Output:
%     cropped - uint8 RGB image with black borders removed

try
    gray = rgb2gray(img);
    % Threshold: pixels brighter than 10/255 are retinal content
    mask = gray > 10;
    % Find bounding box of the retinal content
    rows = any(mask, 2);
    cols = any(mask, 1);
    rMin = find(rows, 1, 'first');
    rMax = find(rows, 1, 'last');
    cMin = find(cols, 1, 'first');
    cMax = find(cols, 1, 'last');

    if isempty(rMin) || isempty(cMin)
        cropped = img;
        return;
    end

    % Add a small padding margin (5px) if possible
    pad = 5;
    rMin = max(1, rMin - pad);
    rMax = min(size(img,1), rMax + pad);
    cMin = max(1, cMin - pad);
    cMax = min(size(img,2), cMax + pad);

    cropped = img(rMin:rMax, cMin:cMax, :);
catch
    cropped = img;
end
end
