function result = runScreeningPipeline(img, cfg)
% runScreeningPipeline  Orchestrates the full DR screening pipeline.
%
%   result = runScreeningPipeline(img, cfg)
%
%   Input:
%     img  - uint8 RGB retinal image (any size)
%     cfg  - RGConfig() struct
%
%   Output:
%     result - struct with:
%       .quality        - image quality assessment
%       .enhanced       - enhanced image (uint8 RGB)
%       .enhanceSteps   - preprocessing step images
%       .structures     - retinal structure detections
%       .lesions        - lesion detections
%       .grading        - DR grade and confidence
%       .pipeline       - processing log (timing per stage)
%       .error          - empty string or error message
%
%   Pipeline stages:
%     1. Image quality assessment
%     2. Image enhancement (if gradable)
%     3. Retinal structure detection
%     4. Lesion detection
%     5. DR severity grading
%
%   If image quality is UNGRADABLE, stages 2–5 are skipped.

if nargin < 2, cfg = RGConfig(); end

result.quality      = [];
result.enhanced     = img;
result.enhanceSteps = struct();
result.structures   = [];
result.lesions      = [];
result.grading      = [];
result.pipeline     = {};
result.error        = '';

t0 = tic;

try
    % ================================================================
    % STAGE 1 — Image Quality Assessment
    % ================================================================
    logStage(result, 'Stage 1: Image quality assessment', tic);
    result.quality = assessImageQuality(img, cfg);
    result.pipeline{end+1} = sprintf('Quality assessment: %.1f s — Score %d/100 — %s', ...
        toc(t0), result.quality.score, result.quality.status);

    % If ungradable, stop pipeline
    if ~result.quality.gradable
        result.error = sprintf('Image quality insufficient (%s). %s', ...
            result.quality.reason, result.quality.recommendation);
        return;
    end

    % ================================================================
    % STAGE 2 — Image Enhancement
    % ================================================================
    t1 = tic;
    [result.enhanced, result.enhanceSteps] = enhanceRetinalImage(img, cfg);
    result.pipeline{end+1} = sprintf('Enhancement: %.1f s', toc(t1));

    % ================================================================
    % STAGE 3 — Retinal Structure Detection
    % ================================================================
    t2 = tic;
    result.structures = detectStructures(result.enhanced, cfg);
    result.pipeline{end+1} = sprintf('Structure detection: %.1f s', toc(t2));

    % ================================================================
    % STAGE 4 — Lesion Detection
    % ================================================================
    t3 = tic;
    result.lesions = detectLesions(result.enhanced, cfg, result.structures);
    result.pipeline{end+1} = sprintf('Lesion detection: %.1f s', toc(t3));

    % ================================================================
    % STAGE 5 — DR Severity Grading
    % ================================================================
    t4 = tic;
    result.grading = gradeDR(result.enhanced, result.lesions, result.quality, cfg);
    result.pipeline{end+1} = sprintf('DR grading: %.1f s', toc(t4));

    result.pipeline{end+1} = sprintf('Total: %.1f s', toc(t0));

catch ME
    result.error = ['Pipeline error: ' ME.message];
    result.pipeline{end+1} = ['ERROR: ' ME.message];
end
end

function logStage(~, msg, ~)
% Placeholder logger (extend later for file logging)
% disp(['[PIPELINE] ' msg]);
end
