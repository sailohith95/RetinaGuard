%% runONNXModel.m
%  Run inference using the APTOS-trained ONNX model.
%  Called by gradeDR.m when the ONNX model file is present.
%
%  result = runONNXModel(img, cfg)
%
%  Input:
%    img  - uint8 RGB retinal image (any size)
%    cfg  - RGConfig() struct
%
%  Output:
%    result - struct with grade, confidence, probabilities, label, etc.
%
%  REQUIREMENTS:
%    MATLAB R2023b or later (for importNetworkFromONNX)
%    Deep Learning Toolbox
%
%  NOTES:
%    The ONNX model is trained with ImageNet normalization.
%    Preprocessing in this function MUST match training/inference/predict.py.

function result = runONNXModel(img, cfg)
if nargin < 2, cfg = RGConfig(); end

CLASS_NAMES = {'No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'};
REFERRAL_THRESHOLD = 2;

% ── Build ONNX path ────────────────────────────────────────────────────
rootDir  = cfg.paths.root;
onnxPath = fullfile(rootDir, 'models/aptos_efficientnet/best_model.onnx');

if ~isfile(onnxPath)
    error('ONNX model not found: %s\nRun: python python/deployment/export_model.py', onnxPath);
end

% ── Load network (cached in persistent variable) ───────────────────────
persistent net netPath
if isempty(net) || ~strcmp(netPath, onnxPath)
    fprintf('Loading ONNX model from: %s\n', onnxPath);
    net     = importNetworkFromONNX(onnxPath);
    netPath = onnxPath;
end

% ── Preprocess image ───────────────────────────────────────────────────
% 1. Resize to 224x224
imgResized = imresize(img, [224 224]);

% 2. Convert to double [0,1]
imgDouble = im2double(imgResized);

% 3. ImageNet normalization (must match Python training)
mean_vals = reshape([0.485 0.456 0.406], 1, 1, 3);
std_vals  = reshape([0.229 0.224 0.225], 1, 1, 3);
imgNorm   = (imgDouble - mean_vals) ./ std_vals;

% 4. Rearrange to NCHW (1 x 3 x H x W) for ONNX
%    MATLAB dlarray uses 'SSCB' format; ONNX expects NCHW
imgNHWC  = single(permute(imgNorm, [1 2 3]));  % H x W x C
imgNCHW  = permute(imgNHWC, [3 1 2]);           % C x H x W
inputDL  = dlarray(reshape(imgNCHW, 1, size(imgNCHW,1), size(imgNCHW,2), size(imgNCHW,3)), 'NCSS');

% ── Forward pass ──────────────────────────────────────────────────────
try
    logits = predict(net, inputDL);
catch ME
    % Fallback for different MATLAB toolbox versions
    try
        logits = net(inputDL);
    catch ME2
        error('ONNX inference failed: %s', ME2.message);
    end
end

% ── Softmax → probabilities ───────────────────────────────────────────
logits_vec  = double(extractdata(logits(:)));
probs       = softmax(logits_vec);
[conf, idx] = max(probs);
grade       = idx - 1;  % 0-indexed

% ── Referral text ─────────────────────────────────────────────────────
referralTexts = {...
    'No signs of DR detected. Routine follow-up in 12 months.', ...
    'Mild signs detected. Follow-up in 6-12 months.', ...
    'Moderate NPDR. Ophthalmology evaluation within 3-6 months.', ...
    'Severe NPDR. Urgent ophthalmology referral within 1 month.', ...
    'Proliferative DR. Urgent referral required.'};

% ── Build result struct ───────────────────────────────────────────────
result.grade          = grade;
result.confidence     = conf;
result.probabilities  = probs';
result.label          = CLASS_NAMES{grade+1};
result.shortLabel     = CLASS_NAMES{grade+1};
result.referral       = grade >= REFERRAL_THRESHOLD;
result.referralText   = referralTexts{grade+1};
result.demoMode       = false;
result.modelInfo      = sprintf('APTOS ONNX EfficientNet-B0 — %s', onnxPath);

if conf >= 0.80
    result.confidenceLevel = 'high';
elseif conf >= 0.65
    result.confidenceLevel = 'moderate';
else
    result.confidenceLevel = 'uncertain';
end

% Placeholder heatmap (Grad-CAM not yet available in MATLAB path)
result.heatmap = img;

end


function p = softmax(x)
% Numerically stable softmax
e = exp(x - max(x));
p = e / sum(e);
end
