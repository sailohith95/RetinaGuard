function result = gradeDR(img, lesions, quality, cfg)
% gradeDR  Estimates DR severity grade using a placeholder model.
%
%   result = gradeDR(img, lesions, quality, cfg)
%
%   Input:
%     img      - uint8 RGB enhanced retinal image
%     lesions  - output of detectLesions()
%     quality  - output of assessImageQuality()
%     cfg      - RGConfig() struct
%
%   Output:
%     result.grade          - integer 0–4 (ICDR scale)
%     result.confidence     - scalar 0–1
%     result.probabilities  - 1×5 vector (class probabilities)
%     result.label          - full severity string
%     result.shortLabel     - short label
%     result.referral       - logical (true → refer to ophthalmologist)
%     result.referralText   - recommendation string
%     result.confidenceLevel- 'high' | 'moderate' | 'low'
%     result.heatmap        - uint8 RGB placeholder saliency map
%     result.demoMode       - logical
%
%   DEMO / PLACEHOLDER:
%   This function uses a rule-based heuristic derived from the lesion
%   counts detected by detectLesions().  It is NOT a trained model.
%   When the APTOS-trained MATLAB network is available, replace this
%   function body with network inference while keeping the same interface.

if nargin < 4, cfg = RGConfig(); end

result.demoMode       = true;
result.grade          = 0;
result.confidence     = 0;
result.probabilities  = zeros(1,5);
result.label          = '';
result.shortLabel     = '';
result.referral       = false;
result.referralText   = '';
result.confidenceLevel = 'low';
result.heatmap        = img;

try
    % ================================================================
    % Try ONNX model first (Python-trained EfficientNet-B0 via export)
    % ================================================================
    onnxPath = fullfile(cfg.paths.root, 'models', 'aptos_efficientnet', 'best_model.onnx');
    if isfile(onnxPath)
        try
            result = runONNXModel(img, cfg);
            result.demoMode = false;
            return;
        catch ME
            warning('gradeDR: ONNX inference failed (%s). Trying Python bridge.', ME.message);
        end
    end

    % ================================================================
    % Try Python bridge directly (PyTorch best_model.pt)
    % ================================================================
    ptModelPath = fullfile(cfg.paths.root, 'models', 'aptos_efficientnet', 'best_model.pt');
    if isfile(ptModelPath)
        try
            result = runPythonInference(img, cfg);
            if ~result.demoMode
                return;
            end
        catch ME
            warning('gradeDR: Python bridge inference failed (%s). Falling back.', ME.message);
        end
    end

    % ================================================================
    % Try legacy MATLAB .mat model
    % ================================================================
    drModelPath = cfg.paths.drModel;
    if isfile(drModelPath)
        result = runTrainedDRModel(img, drModelPath, cfg);
        result.demoMode = false;
        return;
    end

    % ================================================================
    % PLACEHOLDER: Rule-based grade estimation
    % ================================================================
    ma  = lesions.microaneurysm;
    ex  = lesions.exudate;
    he  = lesions.hemorrhage;
    nv  = lesions.neovascularization;

    % --- Scoring logic -----------------------------------------------
    % Start from grade 0; advance based on detected lesion burden
    grade = 0;

    if ma.detected
        grade = max(grade, 1);  % Any MA → at least Mild NPDR
    end
    if ex.detected && ma.detected
        grade = max(grade, 2);  % Exudates + MA → Moderate
    end
    if he.detected
        grade = max(grade, 2);  % Any hemorrhage → Moderate+
        if he.count >= 5
            grade = max(grade, 3); % Many hemorrhages → Severe
        end
    end
    if ma.count >= 10 && he.detected
        grade = max(grade, 3);
    end
    if nv.detected
        grade = 4;              % NV → PDR
    end

    % Quality penalty: if borderline image, cap confidence
    qualityFactor = 1.0;
    if strcmp(quality.status, 'BORDERLINE')
        qualityFactor = 0.80;
    end

    % Simulated confidence (heuristic)
    baseConf = 0.78 + rand() * 0.15;
    conf = baseConf * qualityFactor;
    conf = min(0.97, max(0.50, conf));

    % Probability distribution (peaked at predicted grade)
    probs = zeros(1,5);
    probs(grade+1) = conf;
    remaining = 1 - conf;
    for k = 1:5
        if k ~= grade+1
            probs(k) = remaining / 4;
        end
    end

    % ---- Populate result ---------------------------------------------
    result.grade         = grade;
    result.confidence    = conf;
    result.probabilities = probs;
    result.label         = cfg.dr.labels{grade+1};
    result.shortLabel    = cfg.dr.shortLabels{grade+1};
    result.referral      = grade >= cfg.dr.referralThreshold;

    if grade == 0
        result.referralText = 'Routine follow-up. Rescreen in 12 months.';
    elseif grade == 1
        result.referralText = 'Mild NPDR detected. Rescreen in 6–12 months. Optimize glycemic control.';
    elseif grade == 2
        result.referralText = 'Moderate NPDR. Ophthalmology evaluation recommended within 3–6 months.';
    elseif grade == 3
        result.referralText = 'Severe NPDR. Urgent ophthalmology referral required within 1 month.';
    else
        result.referralText = 'Proliferative DR. Urgent referral required. Vision-threatening condition.';
    end

    % Confidence level
    if conf >= cfg.confidence.moderate
        result.confidenceLevel = 'high';
    elseif conf >= cfg.confidence.low
        result.confidenceLevel = 'moderate';
    else
        result.confidenceLevel = 'low';
    end

    % Placeholder saliency map (Gaussian centered on image)
    [rows, cols, ~] = size(img);
    [X, Y] = meshgrid(1:cols, 1:rows);
    sig  = min(rows, cols) * 0.30;
    heat = exp(-((X - cols/2).^2 + (Y - rows/2).^2) / (2*sig^2));
    heat = heat / max(heat(:));
    heatRGB = zeros(rows, cols, 3);
    heatRGB(:,:,1) = heat;
    heatRGB(:,:,2) = heat * 0.3;
    overlay = im2double(img) * 0.55 + heatRGB * 0.45;
    overlay = max(0, min(1, overlay));
    result.heatmap = im2uint8(overlay);

catch ME
    warning('gradeDR: %s', ME.message);
end
end
