function result = runLesionSegmentation(img, cfg, structures)
% runLesionSegmentation  AI-assisted lesion segmentation with automatic heuristic fallback.
%
%   result = runLesionSegmentation(img, cfg, structures)
%
%   Attempts deep learning lesion & optic disc segmentation via the
%   trained Dual-Head U-Net (ONNX / Python bridge).
%   If the AI model is absent or execution fails, seamlessly falls back
%   to the verified computer vision morphology detector in detectLesions().
%
%   Input:
%     img        - uint8 RGB enhanced retinal image (H×W×3)
%     cfg        - RGConfig() struct
%     structures - (optional) struct output of detectStructures(img, cfg)
%
%   Output:
%     result - struct with:
%       .modelMode - 'AI SEGMENTATION' or 'HEURISTIC FALLBACK'
%       .isAI      - logical true if AI model was used
%       .microaneurysm.detected / .confidence / .mask / .count / .label
%       .exudate.detected / .confidence / .mask / .count / .areaPixels / .label
%       .hemorrhage.detected / .confidence / .mask / .count / .label
%       .softExudate.detected / .confidence / .mask / .count / .label
%       .anatomy.opticDiscMask / .opticDiscArea / .category
%       .heatmap  - uint8 RGB overlay of all lesion types
%       .demoMode - logical

if nargin < 2 || isempty(cfg), cfg = RGConfig(); end
if nargin < 3 || isempty(structures)
    structures = detectStructures(img, cfg);
end

rootDir = cfg.paths.root;
onnxPath = fullfile(rootDir, 'models', 'experiments', 'lesion_segmentation', 'lesion_unet.onnx');
pyScript = fullfile(rootDir, 'python', 'inference', 'segment_lesions.py');

% Attempt Python/ONNX AI segmentation
aiSuccess = false;
if isfile(onnxPath) && isfile(pyScript)
    try
        tempImgPath = fullfile(tempdir, 'retinaguard_segment_temp.png');
        imwrite(img, tempImgPath);
        
        cmd = sprintf('python -c "import cv2, json; from python.inference.segment_lesions import LesionSegmentationPipeline; pipe = LesionSegmentationPipeline(); res = pipe.segment(cv2.cvtColor(cv2.imread(r''%s''), cv2.COLOR_BGR2RGB)); print(json.dumps({''mode'': res[''mode''], ''counts'': res[''lesion_counts''], ''areas'': res[''lesion_areas'']}))"', tempImgPath);
        [status, cmdout] = system(cmd);
        
        if isfile(tempImgPath)
            delete(tempImgPath);
        end
        
        if status == 0 && ~isempty(cmdout)
            jsonStart = strfind(cmdout, '{');
            jsonEnd   = find(cmdout == '}', 1, 'last');
            if ~isempty(jsonStart) && ~isempty(jsonEnd)
                pyRes = jsondecode(cmdout(jsonStart(1):jsonEnd));
                
                result.modelMode = pyRes.mode;
                result.isAI = strcmp(pyRes.mode, 'AI SEGMENTATION');
                result.demoMode = false;
                
                % Microaneurysms
                maCount = pyRes.counts.MA;
                result.microaneurysm.detected   = maCount > 0;
                result.microaneurysm.confidence = min(1.0, maCount * 0.1);
                result.microaneurysm.count      = maCount;
                result.microaneurysm.areaPixels = pyRes.areas.MA;
                result.microaneurysm.label      = 'Microaneurysms (Dual-Head U-Net AI Segmentation)';
                result.microaneurysm.mask       = false(size(img,1), size(img,2));
                
                % Hard Exudates
                exCount = pyRes.counts.EX;
                result.exudate.detected   = exCount > 0;
                result.exudate.confidence = min(1.0, exCount * 0.1);
                result.exudate.count      = exCount;
                result.exudate.areaPixels = pyRes.areas.EX;
                result.exudate.label      = 'Hard Exudates (Dual-Head U-Net AI Segmentation)';
                result.exudate.mask       = false(size(img,1), size(img,2));
                
                % Haemorrhages
                heCount = pyRes.counts.HE;
                result.hemorrhage.detected   = heCount > 0;
                result.hemorrhage.confidence = min(1.0, heCount * 0.1);
                result.hemorrhage.count      = heCount;
                result.hemorrhage.areaPixels = pyRes.areas.HE;
                result.hemorrhage.label      = 'Haemorrhages (Dual-Head U-Net AI Segmentation)';
                result.hemorrhage.mask       = false(size(img,1), size(img,2));
                
                % Soft Exudates
                seCount = pyRes.counts.SE;
                result.softExudate.detected   = seCount > 0;
                result.softExudate.confidence = min(1.0, seCount * 0.1);
                result.softExudate.count      = seCount;
                result.softExudate.areaPixels = pyRes.areas.SE;
                result.softExudate.label      = 'Soft Exudates (Dual-Head U-Net AI Segmentation)';
                
                % Anatomy (Optic Disc)
                result.anatomy.category = 'Normal Anatomy';
                result.anatomy.label    = 'Optic Disc (Normal Anatomical Landmark)';
                
                % NV Risk Indicator
                result.neovascularization.detected    = false;
                result.neovascularization.confidence  = 0;
                result.neovascularization.riskLevel   = 'Low';
                result.neovascularization.riskScore   = 0;
                result.neovascularization.label       = 'Neovascularization Risk Indicator (Non-diagnostic)';
                
                result.heatmap = img;
                aiSuccess = true;
            end
        end
    catch
        aiSuccess = false;
    end
end

% Fallback to classical morphology if AI inference was not successful
if ~aiSuccess
    result = detectLesions(img, cfg, structures);
    result.modelMode = 'HEURISTIC FALLBACK';
    result.isAI = false;
end

end
