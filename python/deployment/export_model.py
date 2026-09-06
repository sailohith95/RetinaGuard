"""
RetinaGuard — ONNX Export for MATLAB Deployment
=================================================
Exports the trained PyTorch model to ONNX format for loading in MATLAB.

Usage:
    python deployment/export_model.py
    python deployment/export_model.py --config python/config/config.yaml

Output:
    models/aptos_efficientnet/best_model.onnx

MATLAB loading:
    net = importNetworkFromONNX('models/aptos_efficientnet/best_model.onnx');

The export includes:
  - ONNX model file (opset 17, dynamic batch)
  - MATLAB wrapper function (core/grading/runONNXModel.m)
  - Verification of ONNX forward pass matches PyTorch
"""

import json
import logging
import sys
from pathlib import Path

# Fix Windows console encoding for PyTorch exporter emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).parent
PYTHON_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinaguard.export")


def export_to_onnx(cfg: dict) -> Path:
    """Export trained PyTorch model to ONNX."""
    model_path = PROJECT_ROOT / cfg["paths"]["best_model"]
    onnx_path = PROJECT_ROOT / cfg["paths"]["onnx_model"]
    image_size = cfg["preprocessing"]["image_size"]
    opset = cfg.get("export", {}).get("opset_version", 17)
    dynamic = cfg.get("export", {}).get("dynamic_axes", True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.\n"
            f"Train first: python python/training/train.py"
        )

    logger.info("Loading model from %s", model_path)
    from models.efficientnet_dr import build_model
    device = torch.device("cpu")  # Export on CPU for maximum compatibility
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Dummy input (batch=1, RGB, model's expected input size)
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Verify forward pass
    with torch.no_grad():
        torch_out = model(dummy_input)
    logger.info("PyTorch output shape: %s  |  logits sample: %s",
                tuple(torch_out.shape), torch_out[0].tolist()[:3])

    # Export
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes_dict = None
    if dynamic:
        dynamic_axes_dict = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
    )
    logger.info("ONNX model exported to %s", onnx_path)

    # Verify ONNX model
    _verify_onnx(onnx_path, dummy_input, torch_out)

    # Write MATLAB wrapper
    _write_matlab_wrapper(cfg, onnx_path, image_size)

    return onnx_path


def _verify_onnx(onnx_path: Path, dummy_input: torch.Tensor, torch_out: torch.Tensor):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed.")

        ort_session = ort.InferenceSession(str(onnx_path))
        ort_out = ort_session.run(
            None, {"input": dummy_input.numpy()}
        )[0]

        max_diff = float(np.abs(torch_out.detach().numpy() - ort_out).max())
        logger.info("Max output difference PyTorch vs ONNX: %.2e", max_diff)
        if max_diff < 1e-4:
            logger.info("[OK] ONNX verification passed (max diff < 1e-4)")
        else:
            logger.warning("[WARN] ONNX output differs from PyTorch by %.2e", max_diff)

    except ImportError as e:
        logger.warning("ONNX verification skipped: %s", e)
    except Exception as e:
        logger.warning("ONNX verification failed: %s", e)


def _write_matlab_wrapper(cfg: dict, onnx_path: Path, image_size: int):
    """Write MATLAB function that loads and runs the ONNX model."""
    grading_dir = PROJECT_ROOT / "core" / "grading"
    grading_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = grading_dir / "runONNXModel.m"

    # ImageNet normalization constants
    mean = cfg["preprocessing"].get("mean", [0.485, 0.456, 0.406])
    std  = cfg["preprocessing"].get("std",  [0.229, 0.224, 0.225])

    onnx_rel = str(onnx_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

    matlab_code = f"""%% runONNXModel.m
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

CLASS_NAMES = {{'No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'}};
REFERRAL_THRESHOLD = 2;

% ── Build ONNX path ────────────────────────────────────────────────────
rootDir  = cfg.paths.root;
onnxPath = fullfile(rootDir, '{onnx_rel}');

if ~isfile(onnxPath)
    error('ONNX model not found: %s\\nRun: python python/deployment/export_model.py', onnxPath);
end

% ── Load network (cached in persistent variable) ───────────────────────
persistent net netPath
if isempty(net) || ~strcmp(netPath, onnxPath)
    fprintf('Loading ONNX model from: %s\\n', onnxPath);
    net     = importNetworkFromONNX(onnxPath);
    netPath = onnxPath;
end

% ── Preprocess image ───────────────────────────────────────────────────
% 1. Resize to {image_size}x{image_size}
imgResized = imresize(img, [{image_size} {image_size}]);

% 2. Convert to double [0,1]
imgDouble = im2double(imgResized);

% 3. ImageNet normalization (must match Python training)
mean_vals = reshape([{mean[0]} {mean[1]} {mean[2]}], 1, 1, 3);
std_vals  = reshape([{std[0]} {std[1]} {std[2]}], 1, 1, 3);
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
referralTexts = {{...
    'No signs of DR detected. Routine follow-up in 12 months.', ...
    'Mild signs detected. Follow-up in 6-12 months.', ...
    'Moderate NPDR. Ophthalmology evaluation within 3-6 months.', ...
    'Severe NPDR. Urgent ophthalmology referral within 1 month.', ...
    'Proliferative DR. Urgent referral required.'}};

% ── Build result struct ───────────────────────────────────────────────
result.grade          = grade;
result.confidence     = conf;
result.probabilities  = probs';
result.label          = CLASS_NAMES{{grade+1}};
result.shortLabel     = CLASS_NAMES{{grade+1}};
result.referral       = grade >= REFERRAL_THRESHOLD;
result.referralText   = referralTexts{{grade+1}};
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
"""

    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(matlab_code)
    logger.info("MATLAB wrapper written to %s", wrapper_path)


# ==============================================================================
#  Entry point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export RetinaGuard model to ONNX")
    parser.add_argument("--config",
                        default=str(PYTHON_DIR / "config" / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    onnx_path = export_to_onnx(cfg)
    print(f"\n[OK] ONNX export complete: {onnx_path}")
    print(f"\nIn MATLAB:")
    print(f"  net = importNetworkFromONNX('{onnx_path}');")
    print(f"  result = runONNXModel(img, cfg);")
