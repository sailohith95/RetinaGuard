function cfg = RGConfig()
% RGConfig  Returns the central configuration structure for RetinaGuard.
%
%   cfg = RGConfig()
%
%   Edit this file to change dataset paths, model paths, or threshold
%   values without touching any other source file.

% -------------------------------------------------------------------------
% Paths (relative to the RetinaGuard root folder)
% -------------------------------------------------------------------------
base = fileparts(fileparts(mfilename('fullpath'))); % RetinaGuard root

cfg.paths.root          = base;
cfg.paths.data          = fullfile(base, 'data');
cfg.paths.aptos         = fullfile(base, 'data', 'aptos');
cfg.paths.idrid         = fullfile(base, 'data', 'idrid');
cfg.paths.models        = fullfile(base, 'models');
cfg.paths.drModel       = fullfile(base, 'models', 'dr', 'dr_classifier.mat');
cfg.paths.onnxModel     = fullfile(base, 'models', 'aptos_efficientnet', 'best_model.onnx');
cfg.paths.qualityModel  = fullfile(base, 'models', 'quality', 'quality_model.mat');
cfg.paths.lesionModel   = fullfile(base, 'models', 'lesions', 'lesion_model.mat');
cfg.paths.structModel   = fullfile(base, 'models', 'structures', 'structure_model.mat');
cfg.paths.reports       = fullfile(base, 'reports');
cfg.paths.demo          = fullfile(base, 'demo');
cfg.paths.pythonPredict = fullfile(base, 'python', 'inference', 'predict.py');
cfg.paths.results       = fullfile(base, 'results');

% -------------------------------------------------------------------------
% Image preprocessing
% -------------------------------------------------------------------------
cfg.preprocess.targetSize      = [512 512];   % pixels
cfg.preprocess.claheClipLimit  = 0.02;
cfg.preprocess.claheTileSize   = [8 8];
cfg.preprocess.denoiseMethod   = 'gaussian';  % 'gaussian' | 'median'
cfg.preprocess.denoiseSigma    = 0.5;

% -------------------------------------------------------------------------
% Image quality thresholds
% -------------------------------------------------------------------------
cfg.quality.minScore           = 40;   % below this → ungradable
cfg.quality.borderlineScore    = 60;   % 40–60 → borderline
% Focus
cfg.quality.focusLaplaceMin    = 80;   % variance of Laplacian
% Illumination
cfg.quality.illumMeanMin       = 0.15; % normalized 0-1
cfg.quality.illumMeanMax       = 0.85;
% Field of View (fraction of non-black pixels)
cfg.quality.fovMinFraction     = 0.40;

% -------------------------------------------------------------------------
% DR classification confidence thresholds
% -------------------------------------------------------------------------
cfg.confidence.low             = 0.70; % < 70 % → uncertain
cfg.confidence.moderate        = 0.85; % 70–85 % → moderate

% -------------------------------------------------------------------------
% DR severity labels (ICDR scale 0–4)
% -------------------------------------------------------------------------
cfg.dr.labels = { ...
    'No Diabetic Retinopathy',              ...  % Level 0
    'Mild Non-Proliferative DR',            ...  % Level 1
    'Moderate Non-Proliferative DR',        ...  % Level 2
    'Severe Non-Proliferative DR',          ...  % Level 3
    'Proliferative Diabetic Retinopathy'};       % Level 4

cfg.dr.shortLabels = {'No DR','Mild NPDR','Moderate NPDR','Severe NPDR','PDR'};

cfg.dr.referralThreshold = 2; % Grade ≥ 2 → refer to ophthalmologist

% -------------------------------------------------------------------------
% Application settings
% -------------------------------------------------------------------------
cfg.app.name        = 'RetinaGuard';
cfg.app.version     = '1.0.0-prototype';
cfg.app.demoMode    = true;  % overridden at runtime if real models exist
cfg.app.supportedFmt = {'*.jpg','*.jpeg','*.png','*.tif','*.tiff'};

end
