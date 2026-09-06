function result = assessImageQuality(img, cfg)
% assessImageQuality  Evaluates the quality of a retinal fundus image.
%
%   result = assessImageQuality(img, cfg)
%
%   Input:
%     img  - uint8 or double RGB image (H×W×3)
%     cfg  - RGConfig() struct
%
%   Output:
%     result  - struct with fields:
%       .score          - overall quality score (0–100)
%       .status         - 'GOOD' | 'BORDERLINE' | 'UNGRADABLE'
%       .focus          - 'Good' | 'Acceptable' | 'Poor'
%       .focusScore     - numeric (Laplacian variance)
%       .illumination   - 'Good' | 'Acceptable' | 'Poor'
%       .illuminScore   - numeric (mean pixel intensity, 0–1)
%       .fieldOfView    - 'Good' | 'Acceptable' | 'Poor'
%       .fovScore       - fraction of in-field pixels
%       .contrast       - 'Good' | 'Acceptable' | 'Poor'
%       .contrastScore  - numeric (std of green channel)
%       .retinalVisibility - 'Good' | 'Acceptable' | 'Poor'
%       .reason         - string (empty if GOOD)
%       .recommendation - string (empty if GOOD)
%       .gradable       - logical

if nargin < 2, cfg = RGConfig(); end

% Ensure double
if ~isa(img,'double')
    imgD = im2double(img);
else
    imgD = img;
end

result.score        = 0;
result.status       = 'UNGRADABLE';
result.focus        = 'Poor';
result.focusScore   = 0;
result.illumination = 'Poor';
result.illuminScore = 0;
result.fieldOfView  = 'Poor';
result.fovScore     = 0;
result.contrast     = 'Poor';
result.contrastScore = 0;
result.retinalVisibility = 'Poor';
result.reason       = '';
result.recommendation = '';
result.gradable     = false;

try
    % ---- Focus assessment (Laplacian variance of green channel) ----------
    grayImg = rgb2gray(imgD);
    lap     = fspecial('laplacian', 0.2);
    lapImg  = imfilter(grayImg, lap, 'replicate');
    focusVar = var(lapImg(:)) * 1e4;  % scale to 0–300ish range
    result.focusScore = focusVar;

    focusThresh = cfg.quality.focusLaplaceMin;
    if focusVar >= focusThresh
        result.focus = 'Good';
        focusComponent = 30;
    elseif focusVar >= focusThresh * 0.5
        result.focus = 'Acceptable';
        focusComponent = 20;
    else
        result.focus = 'Poor';
        focusComponent = 5;
    end

    % ---- Illumination assessment ----------------------------------------
    greenCh = imgD(:,:,2);
    meanIntensity = mean(greenCh(:));
    result.illuminScore = meanIntensity;

    illumMin = cfg.quality.illumMeanMin;
    illumMax = cfg.quality.illumMeanMax;
    if meanIntensity >= illumMin && meanIntensity <= illumMax
        result.illumination = 'Good';
        illumComponent = 25;
    elseif meanIntensity >= illumMin*0.6 && meanIntensity <= illumMax*1.1
        result.illumination = 'Acceptable';
        illumComponent = 15;
    else
        result.illumination = 'Poor';
        illumComponent = 3;
    end

    % ---- Field of View assessment ----------------------------------------
    % Estimate retinal region as non-very-dark pixels
    grayUint = rgb2gray(img);
    retinalMask = grayUint > 10;
    fovFrac = sum(retinalMask(:)) / numel(retinalMask);
    result.fovScore = fovFrac;

    fovMin = cfg.quality.fovMinFraction;
    if fovFrac >= fovMin
        result.fieldOfView = 'Good';
        fovComponent = 25;
    elseif fovFrac >= fovMin * 0.7
        result.fieldOfView = 'Acceptable';
        fovComponent = 15;
    else
        result.fieldOfView = 'Poor';
        fovComponent = 3;
    end

    % ---- Contrast assessment --------------------------------------------
    stdIntensity = std(greenCh(:));
    result.contrastScore = stdIntensity;

    if stdIntensity >= 0.10
        result.contrast = 'Good';
        contrastComponent = 10;
    elseif stdIntensity >= 0.05
        result.contrast = 'Acceptable';
        contrastComponent = 6;
    else
        result.contrast = 'Poor';
        contrastComponent = 1;
    end

    % ---- Retinal visibility (combined proxy) ----------------------------
    visScore = (focusVar / focusThresh) * 0.5 + fovFrac * 0.5;
    if visScore >= 0.75
        result.retinalVisibility = 'Good';
    elseif visScore >= 0.4
        result.retinalVisibility = 'Acceptable';
    else
        result.retinalVisibility = 'Poor';
    end

    % ---- Overall score ---------------------------------------------------
    result.score = round(focusComponent + illumComponent + fovComponent + contrastComponent);
    result.score = min(100, max(0, result.score));

    % ---- Status determination -------------------------------------------
    if result.score >= cfg.quality.borderlineScore
        result.status  = 'GOOD';
        result.gradable = true;
    elseif result.score >= cfg.quality.minScore
        result.status  = 'BORDERLINE';
        result.gradable = true;
    else
        result.status  = 'UNGRADABLE';
        result.gradable = false;
    end

    % ---- Reason/recommendation if not good ------------------------------
    if strcmp(result.focus, 'Poor')
        result.reason = 'Image appears blurry or out of focus.';
        result.recommendation = 'Recapture: ensure camera is in focus and patient is steady.';
    elseif strcmp(result.illumination, 'Poor')
        if meanIntensity < illumMin
            result.reason = 'Image is too dark.';
            result.recommendation = 'Recapture: increase illumination or adjust camera exposure.';
        else
            result.reason = 'Image is overexposed.';
            result.recommendation = 'Recapture: reduce illumination or adjust camera exposure.';
        end
    elseif strcmp(result.fieldOfView, 'Poor')
        result.reason = 'Retinal field coverage is insufficient.';
        result.recommendation = 'Recapture: center the retina and ensure full field coverage.';
    end

catch ME
    result.reason = ['Quality assessment error: ' ME.message];
    result.recommendation = 'Check that the image is a valid retinal fundus image.';
end

result.gradable = result.score >= cfg.quality.minScore;
end
