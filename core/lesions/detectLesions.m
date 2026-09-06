function result = detectLesions(img, cfg, structures)
% detectLesions  Detects diabetic retinopathy lesions in a fundus image.
%
%   result = detectLesions(img, cfg, structures)
%
%   Detects:
%     - Microaneurysm candidates (MA)
%     - Hard exudate candidates (with Optic Disc exclusion)
%     - Hemorrhage candidates
%     - Neovascularization Risk Indicator (based on measurable vessel characteristics)
%
%   Input:
%     img        - uint8 RGB enhanced retinal image (H×W×3)
%     cfg        - RGConfig() struct
%     structures - (optional) struct output of detectStructures(img, cfg)
%
%   Output:
%     result - struct with:
%       .microaneurysm.detected / .confidence / .mask / .count / .label
%       .exudate.detected / .confidence / .mask / .count / .areaPixels / .label
%       .hemorrhage.detected / .confidence / .mask / .count / .label
%       .neovascularization.detected / .confidence / .riskLevel / .riskScore / .label
%       .heatmap  - uint8 RGB overlay of all lesion types
%       .demoMode - logical

if nargin < 2 || isempty(cfg), cfg = RGConfig(); end
if nargin < 3 || isempty(structures)
    structures = detectStructures(img, cfg);
end

result.demoMode = true;

% ---- Safe defaults --------------------------------------------------
result.microaneurysm.detected    = false;
result.microaneurysm.confidence  = 0;
result.microaneurysm.mask        = false(size(img,1), size(img,2));
result.microaneurysm.count       = 0;
result.microaneurysm.label       = 'Microaneurysm Candidates (Computer Vision)';

result.exudate.detected          = false;
result.exudate.confidence        = 0;
result.exudate.mask              = false(size(img,1), size(img,2));
result.exudate.count             = 0;
result.exudate.areaPixels        = 0;
result.exudate.label             = 'Exudate Candidates (Optic-Disc Aware)';

result.hemorrhage.detected       = false;
result.hemorrhage.confidence     = 0;
result.hemorrhage.mask           = false(size(img,1), size(img,2));
result.hemorrhage.count          = 0;
result.hemorrhage.label          = 'Hemorrhage Candidates (Computer Vision)';

result.neovascularization.detected    = false;
result.neovascularization.confidence  = 0;
result.neovascularization.riskLevel   = 'Low';
result.neovascularization.riskScore   = 0;
result.neovascularization.label       = 'Neovascularization Risk Indicator (Non-diagnostic)';
result.neovascularization.mask        = false(size(img,1), size(img,2));

result.heatmap = img;

try
    [rows, cols, ~] = size(img);
    imgD = im2double(img);
    r    = imgD(:,:,1);
    g    = imgD(:,:,2);
    b    = imgD(:,:,3);

    % Retinal mask (non-black pixels)
    gray    = rgb2gray(imgD);
    retMask = gray > 0.05;

    % Retrieve optic disc mask to avoid false positive exudates on the optic disc
    if isfield(structures, 'opticDisc') && isfield(structures.opticDisc, 'mask') && any(structures.opticDisc.mask(:))
        odMask = structures.opticDisc.mask;
        odDilated = imdilate(odMask, strel('disk', max(2, round(structures.opticDisc.radius * 0.25))));
    else
        % Fallback disc mask from brightest peak
        blurred = imgaussfilt(gray, 15);
        [~, maxIdx] = max(blurred(:));
        [odRow, odCol] = ind2sub([rows cols], maxIdx);
        odR = round(min(rows, cols) * 0.07);
        [Ygrid, Xgrid] = ndgrid(1:rows, 1:cols);
        odDilated = ((Ygrid - odRow).^2 + (Xgrid - odCol).^2) <= ((odR * 1.25)^2);
    end

    % Retrieve blood vessel mask
    if isfield(structures, 'vessels') && isfield(structures.vessels, 'mask')
        vesselMask = structures.vessels.mask;
    else
        vesselMask = false(rows, cols);
    end
    dilatedVessels = imdilate(vesselMask, strel('disk', 2));

    % ================================================================
    % 1. HARD EXUDATES — bright yellowish-white regions
    % CRITICAL: Optic disc region is EXCLUDED
    % ================================================================
    exuRaw = (r > 0.65) & (g > 0.55) & (b < 0.60) & (r + g > 1.25) & retMask & ~odDilated;
    exuMask = bwareaopen(exuRaw, 6);   % remove single-pixel noise
    exuMask = imdilate(exuMask, strel('disk', 1));
    exuCC   = bwconncomp(exuMask);
    exuCount = exuCC.NumObjects;
    exuArea  = sum(exuMask(:));

    result.exudate.mask       = exuMask;
    result.exudate.count      = exuCount;
    result.exudate.areaPixels = exuArea;
    result.exudate.detected   = exuCount > 0;
    result.exudate.confidence = min(0.95, 0.50 + exuCount * 0.05);

    % ================================================================
    % 2. MICROANEURYSMS — small dark circular blobs on green channel
    % ================================================================
    invG = 1.0 - g;
    tophat = imtophat(invG, strel('disk', 4));
    maRaw = (tophat > 0.07) & retMask & ~dilatedVessels & ~odDilated;
    maRaw = bwareaopen(maRaw, 3);
    
    % Size filter: 3 to 60 pixels
    maProps = regionprops(maRaw, 'PixelIdxList', 'Area');
    maMask  = false(rows, cols);
    maCount = 0;
    for k = 1:length(maProps)
        if maProps(k).Area >= 3 && maProps(k).Area <= 65
            maMask(maProps(k).PixelIdxList) = true;
            maCount = maCount + 1;
        end
    end

    result.microaneurysm.mask       = maMask;
    result.microaneurysm.count      = maCount;
    result.microaneurysm.detected   = maCount > 0;
    result.microaneurysm.confidence = min(0.92, 0.45 + maCount * 0.04);

    % ================================================================
    % 3. HEMORRHAGES — larger dark intra-retinal lesions
    % ================================================================
    heRaw = (g < 0.28) & (r > 0.22) & (b < 0.25) & retMask & ~dilatedVessels & ~odDilated;
    heRaw = bwareaopen(heRaw, 40);
    heProps = regionprops(heRaw, 'PixelIdxList', 'Area');
    heMask  = false(rows, cols);
    heCount = 0;
    for k = 1:length(heProps)
        if heProps(k).Area >= 40 && heProps(k).Area <= 2500
            heMask(heProps(k).PixelIdxList) = true;
            heCount = heCount + 1;
        end
    end

    result.hemorrhage.mask       = heMask;
    result.hemorrhage.count      = heCount;
    result.hemorrhage.detected   = heCount > 0;
    result.hemorrhage.confidence = min(0.92, 0.50 + heCount * 0.05);

    % ================================================================
    % 4. NEOVASCULARIZATION RISK INDICATOR (Measurable Vessel Dynamics)
    % ================================================================
    % Measure peri-papillary vessel density & fine-branch proliferation
    totalRetinal = max(1, sum(retMask(:)));
    overallVesselDensity = (sum(vesselMask(:) & retMask(:)) / totalRetinal) * 100;

    % Peri-papillary zone (within 2.5x disc radius)
    if isfield(structures, 'opticDisc') && isfield(structures.opticDisc, 'centroid')
        odCenter = structures.opticDisc.centroid;
        odR = structures.opticDisc.radius;
        [Yg, Xg] = ndgrid(1:rows, 1:cols);
        periZone = ((Yg - odCenter(1)).^2 + (Xg - odCenter(2)).^2) <= ((odR * 2.5)^2) & ~odDilated & retMask;
        periVesselPixels = sum(vesselMask(:) & periZone(:));
        periZonePixels   = max(1, sum(periZone(:)));
        periDensity      = (periVesselPixels / periZonePixels) * 100;
    else
        periDensity = overallVesselDensity;
    end

    % Fine vessel proliferation ratio
    skelVessels = bwmorph(vesselMask, 'skel', Inf);
    branchPoints = bwmorph(skelVessels, 'branchpoints');
    branchCount = sum(branchPoints(:));
    branchDensity = (branchCount / max(1, sum(skelVessels(:)))) * 100;

    % Risk score formula (0 to 100)
    nvScore = min(100, max(0, round((periDensity * 2.2) + (branchDensity * 3.5))));
    if nvScore >= 65
        nvRisk = 'High';
        nvDetected = true;
    elseif nvScore >= 40
        nvRisk = 'Moderate';
        nvDetected = false;
    else
        nvRisk = 'Low';
        nvDetected = false;
    end

    result.neovascularization.riskLevel            = nvRisk;
    result.neovascularization.riskScore            = nvScore;
    result.neovascularization.peripapillaryDensity = round(periDensity, 2);
    result.neovascularization.branchingScore       = round(branchDensity, 2);
    result.neovascularization.detected             = nvDetected;
    result.neovascularization.confidence           = round(nvScore / 100, 2);
    result.neovascularization.mask                 = branchPoints;

    % ================================================================
    % 5. MULTI-LESION CLINICAL OVERLAY
    % ================================================================
    heatmap = imgD;
    % Exudates -> Yellow tint
    exuDil = imdilate(exuMask, strel('disk', 1));
    for c = 1:3
        ch = heatmap(:,:,c);
        if c == 1, tint = 1.0; elseif c == 2, tint = 0.88; else, tint = 0.0; end
        ch(exuDil) = 0.6 * ch(exuDil) + 0.4 * tint;
        heatmap(:,:,c) = ch;
    end

    % Hemorrhages -> Deep red tint
    heDil = imdilate(heMask, strel('disk', 1));
    for c = 1:3
        ch = heatmap(:,:,c);
        if c == 1, tint = 0.9; else, tint = 0.1; end
        ch(heDil) = 0.5 * ch(heDil) + 0.5 * tint;
        heatmap(:,:,c) = ch;
    end

    % Microaneurysms -> Magenta dots
    maDil = imdilate(maMask, strel('disk', 2));
    for c = 1:3
        ch = heatmap(:,:,c);
        if c == 1, tint = 0.95; elseif c == 2, tint = 0.05; else, tint = 0.95; end
        ch(maDil) = 0.4 * ch(maDil) + 0.6 * tint;
        heatmap(:,:,c) = ch;
    end

    result.heatmap = im2uint8(heatmap);

catch ME
    warning('detectLesions: %s', ME.message);
end
end
