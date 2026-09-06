%% RetinaGuard — Launch Script
%
%   Run this script to start the RetinaGuard DR Screening application.
%
%   Usage (from MATLAB command window):
%       run('RetinaGuard/launch.m')
%   or:
%       cd RetinaGuard; launch
%
%   Requirements:
%       MATLAB R2021a or later
%       Image Processing Toolbox
%       Computer Vision Toolbox (optional, for future features)
%       Deep Learning Toolbox (optional, for real model inference)

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  RetinaGuard — AI-Assisted Diabetic Retinopathy Screening\n');
fprintf('  SIH 2026 Prototype   |   v1.0.0\n');
fprintf('=================================================================\n\n');

% Add RetinaGuard root and all subdirectories to path
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(thisDir));

% Check MATLAB version
mver = ver('MATLAB');
fprintf('MATLAB version: %s\n', mver.Release);

% Check toolboxes
tbRequired = {'Image Processing Toolbox'};
tbOptional = {'Computer Vision Toolbox', 'Deep Learning Toolbox', ...
              'Statistics and Machine Learning Toolbox'};

fprintf('\nChecking required toolboxes:\n');
allOk = true;
for i = 1:numel(tbRequired)
    if ~isempty(ver(tbRequired{i}))
        fprintf('  [OK] %s\n', tbRequired{i});
    else
        fprintf('  [MISSING] %s — REQUIRED\n', tbRequired{i});
        allOk = false;
    end
end

fprintf('\nChecking optional toolboxes:\n');
for i = 1:numel(tbOptional)
    if ~isempty(ver(tbOptional{i}))
        fprintf('  [OK] %s\n', tbOptional{i});
    else
        fprintf('  [--] %s — not installed (optional)\n', tbOptional{i});
    end
end

if ~allOk
    error('RetinaGuard requires the Image Processing Toolbox. Please install it.');
end

% Check config
cfg = RGConfig();
fprintf('\nConfiguration:\n');
fprintf('  Data directory:    %s\n', cfg.paths.data);
fprintf('  Reports directory: %s\n', cfg.paths.reports);
fprintf('  DR model path:     %s\n', cfg.paths.drModel);

% Determine mode
if isfile(cfg.paths.drModel)
    fprintf('\n  Mode: TRAINED MODEL\n');
else
    fprintf('\n  Mode: DEMO (no trained model found)\n');
    fprintf('  Demo mode is active. Results are not medically validated.\n');
end

% Ensure reports directory exists
if ~isfolder(cfg.paths.reports)
    mkdir(cfg.paths.reports);
    fprintf('  Created reports directory: %s\n', cfg.paths.reports);
end

fprintf('\nLaunching RetinaGuard application...\n\n');

% Launch app
app = RetinaGuardApp();
