function result = runPythonInference(img, cfg)
% runPythonInference  Inference adapter: calls RetinaGuard Python predictDR API
%
%   result = runPythonInference(img, cfg)
%
%   This function serves as a robust bridge between MATLAB and the
%   Python trained model. It saves the enhanced image to a temporary
%   cache file, invokes the Python predictDR() CLI, and parses the
%   returned JSON prediction.
%
%   Input:
%     img  - uint8 RGB retinal image
%     cfg  - RGConfig() struct
%
%   Output:
%     result - struct with fields:
%       .grade, .confidence, .probabilities, .label, .shortLabel,
%       .referral, .referralText, .confidenceLevel, .heatmap, .demoMode

if nargin < 2, cfg = RGConfig(); end

CLASS_NAMES = {'No DR','Mild NPDR','Moderate NPDR','Severe NPDR','PDR'};
REFERRAL_THRESHOLD = cfg.dr.referralThreshold;

tempImgPath  = fullfile(tempdir, 'retinaguard_inference_temp.png');
predictScript = fullfile(cfg.paths.root, 'python', 'inference', 'predict.py');

% Write image to temp file
imwrite(img, tempImgPath);

% Call python script with --json and --gradcam flags
cmd = sprintf('python "%s" "%s" --json --gradcam', predictScript, tempImgPath);
[status, cmdout] = system(cmd);

% Clean up temp image
if isfile(tempImgPath)
    delete(tempImgPath);
end

if status ~= 0 || isempty(cmdout)
    error('runPythonInference: execution failed. Output: %s', cmdout);
end

% Parse JSON from command output (extract text between { and })
jsonStart = strfind(cmdout, '{');
jsonEnd   = find(cmdout == '}', 1, 'last');

if isempty(jsonStart) || isempty(jsonEnd)
    error('runPythonInference: No valid JSON found in output: %s', cmdout);
end

jsonStr = cmdout(jsonStart(1):jsonEnd);
pyRes   = jsondecode(jsonStr);

% Build standard RetinaGuard result struct
grade = pyRes.grade;
conf  = pyRes.confidence;

result.grade          = grade;
result.confidence     = conf;
result.probabilities  = reshape(pyRes.probabilities, 1, 5);
result.label          = cfg.dr.labels{grade + 1};
result.shortLabel     = CLASS_NAMES{grade + 1};
result.referral       = grade >= REFERRAL_THRESHOLD;
result.referralText   = pyRes.referral_text;
result.confidenceLevel= pyRes.confidence_level;
result.demoMode       = pyRes.demo_mode;
result.modelInfo      = pyRes.model_info;

% Load Grad-CAM heatmap if generated
if isfield(pyRes, 'gradcam_path') && ~isempty(pyRes.gradcam_path) && isfile(pyRes.gradcam_path)
    try
        result.heatmap = imread(pyRes.gradcam_path);
        delete(pyRes.gradcam_path);
    catch
        result.heatmap = img;
    end
else
    result.heatmap = img;
end

end
