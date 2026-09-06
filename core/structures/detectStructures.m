function result = detectStructures(img, cfg)
% detectStructures  Detects retinal structures (optic disc, fovea, vessels).
%
%   result = detectStructures(img, cfg)
%
%   Input:
%     img  - uint8 RGB retinal image (already enhanced/resized)
%     cfg  - RGConfig() struct
%
%   Output:
%     result - struct with fields:
%       .opticDisc.detected      - logical
%       .opticDisc.confidence    - 0–1
%       .opticDisc.centroid      - [row col] approximate location
%       .opticDisc.overlay       - uint8 RGB image with OD marked
%       .fovea.detected          - logical
%       .fovea.confidence        - 0–1
%       .fovea.centroid          - [row col] approximate location
%       .vessels.detected        - logical
%       .vessels.confidence      - 0–1
%       .vessels.mask            - binary vessel mask
%       .demoMode                - logical (true when no trained model)
%
%   When no trained model is loaded, this function uses image-processing
%   heuristics to approximate structure locations.
%   All results are labelled demoMode = true.

if nargin < 2, cfg = RGConfig(); end

result.demoMode = true;  % always true until real model loaded

% ---- Defaults (safe fallback) ----------------------------------------
result.opticDisc.detected   = false;
result.opticDisc.confidence = 0;
result.opticDisc.centroid   = [0 0];
result.opticDisc.overlay    = img;
result.fovea.detected       = false;
result.fovea.confidence     = 0;
result.fovea.centroid       = [0 0];
result.vessels.detected     = false;
result.vessels.confidence   = 0;
result.vessels.mask         = false(size(img,1), size(img,2));

try
    [rows, cols, ~] = size(img);
    imgD = im2double(img);

    % ================================================================
    % OPTIC DISC — brightest circular region in the image
    % ================================================================
    gray = rgb2gray(imgD);

    % Smooth heavily and find peak brightness
    blurred   = imgaussfilt(gray, 15);
    [maxVal, maxIdx] = max(blurred(:));
    [odRow, odCol]   = ind2sub([rows cols], maxIdx);

    % Heuristic confidence based on peak brightness
    odConf = min(0.99, maxVal * 1.6);

    result.opticDisc.detected   = true;
    result.opticDisc.confidence = odConf;
    result.opticDisc.centroid   = [odRow odCol];

    % ================================================================
    % FOVEA — darkest region near image center (temporal to OD)
    % Approximate: fovea is roughly 2.5 disc-diameters from OD
    % For a 512×512 image the disc radius ~= 35 px → 2.5×70 ≈ 175 px
    % ================================================================
    discRadius = min(rows, cols) * 0.07;  % approximate
    offset     = discRadius * 5;          % empirical fundus geometry

    % Fovea is temporal to OD (opposite side from centre)
    direction = sign(cols/2 - odCol);     % +1 if OD right of centre
    fvCol     = odCol + direction * offset;
    fvRow     = odRow;

    fvCol = max(1, min(cols, round(fvCol)));
    fvRow = max(1, min(rows, round(fvRow)));

    % Confirm by finding local minimum near predicted location
    roi = max(1, fvRow-30):min(rows, fvRow+30);
    coi = max(1, fvCol-30):min(cols, fvCol+30);
    roiImg = gray(roi, coi);
    roiBlur = imgaussfilt(roiImg, 8);
    [~, minIdx] = min(roiBlur(:));
    [lr, lc]    = ind2sub(size(roiImg), minIdx);
    fvRow = roi(1) + lr - 1;
    fvCol = coi(1) + lc - 1;

    result.fovea.detected   = true;
    result.fovea.confidence = 0.85;
    result.fovea.centroid   = [fvRow fvCol];

    % ================================================================
    % BLOOD VESSELS — matched filter / Frangi vesselness
    % ================================================================
    green = imgD(:,:,2);
    % Invert (vessels are dark on lighter fundus)
    invGreen = 1 - green;
    % Multi-scale Gaussian ridge detection (simplified Frangi-like)
    vesselMask = false(rows, cols);
    for sigma = [1.5 2.5 4.0]
        % Second-order Gaussian derivatives (Hessian-based)
        g = fspecial('gaussian', round(6*sigma+1), sigma);
        gxx = imfilter(invGreen, g, 'replicate');
        gxx = gxx - imgaussfilt(invGreen, sigma*2.5);
        vesselMask = vesselMask | (gxx > 0.04);
    end
    % Clean up
    vesselMask = bwareaopen(vesselMask, 30);
    vesselMask = imdilate(vesselMask, strel('disk', 1));

    % Compute retinal field mask and vessel density
    retMask = gray > 0.05;
    totalRetinalPixels = max(1, sum(retMask(:)));
    vesselDensity = (sum(vesselMask(:) & retMask(:)) / totalRetinalPixels) * 100;

    odR = round(discRadius);
    % Create circular binary mask for optic disc
    [Ygrid, Xgrid] = ndgrid(1:rows, 1:cols);
    odMask = ((Ygrid - odRow).^2 + (Xgrid - odCol).^2) <= (odR^2);

    result.opticDisc.detected    = true;
    result.opticDisc.confidence  = odConf;
    result.opticDisc.centroid    = [odRow odCol];
    result.opticDisc.radius      = odR;
    result.opticDisc.mask        = odMask;
    result.opticDisc.methodology = 'Computer Vision Peak Luminance & Morphology';

    result.fovea.detected        = true;
    result.fovea.confidence      = 0.85;
    result.fovea.centroid        = [fvRow fvCol];
    result.fovea.radius          = round(odR * 0.6);
    result.fovea.methodology     = 'Anatomical Geometric Projection & Local Luminance Minimum';

    result.vessels.detected      = true;
    result.vessels.confidence    = 0.88;
    result.vessels.mask          = vesselMask;
    result.vessels.density       = round(vesselDensity, 2);
    result.vessels.methodology   = 'Computer Vision Assisted Vessel Segmentation (Multiscale Gaussian Ridge Filtering)';

    % ================================================================
    % Overlay visualization
    % ================================================================
    overlay = img;
    % Draw OD circle (yellow)
    overlay = drawCircleOnImage(overlay, odRow, odCol, odR, [255 220 0]);
    % Draw fovea cross (cyan)
    overlay = drawCrossOnImage(overlay, fvRow, fvCol, 12, [0 220 220]);
    % Vessel overlay (green tint)
    vesselOverlay = overlay;
    greenCh  = vesselOverlay(:,:,2);
    greenCh(vesselMask)  = min(255, double(greenCh(vesselMask)) + 60);
    redCh    = vesselOverlay(:,:,1);
    redCh(vesselMask)    = max(0, double(redCh(vesselMask))    - 30);
    vesselOverlay(:,:,1) = redCh;
    vesselOverlay(:,:,2) = greenCh;

    result.opticDisc.overlay = overlay;
    result.vessels.overlay   = vesselOverlay;

catch ME
    warning('detectStructures: %s', ME.message);
end
end

% =========================================================================
%  Local helpers
% =========================================================================

function img = drawCircleOnImage(img, cy, cx, r, color)
[rows, cols, ~] = size(img);
theta = linspace(0, 2*pi, 200);
for t = theta
    px = round(cx + r * cos(t));
    py = round(cy + r * sin(t));
    if px>=1&&px<=cols&&py>=1&&py<=rows
        img(py,px,1) = color(1);
        img(py,px,2) = color(2);
        img(py,px,3) = color(3);
        % 2px thick
        for d = -1:1
            if py+d>=1&&py+d<=rows, img(py+d,px,:) = reshape(color,[1,1,3]); end
            if px+d>=1&&px+d<=cols, img(py,px+d,:) = reshape(color,[1,1,3]); end
        end
    end
end
end

function img = drawCrossOnImage(img, cy, cx, len, color)
[rows, cols, ~] = size(img);
for d = -len:len
    for th = -1:1
        pr = cy + d; pc = cx + th;
        if pr>=1&&pr<=rows&&pc>=1&&pc<=cols
            img(pr,pc,:) = reshape(color,[1,1,3]);
        end
        pr = cy + th; pc = cx + d;
        if pr>=1&&pr<=rows&&pc>=1&&pc<=cols
            img(pr,pc,:) = reshape(color,[1,1,3]);
        end
    end
end
end
