# RetinaGuard — Model Deployment & Export Guide

## Overview

RetinaGuard employs a dual-deployment architecture designed specifically for the SIH 2026 problem statement:

1. **Python Primary AI Engine**: Model training, transfer learning (EfficientNet-B0), Grad-CAM explainability, and evaluation.
2. **ONNX Export for MATLAB**: Standardized open format that directly imports into MATLAB Deep Learning Toolbox.
3. **MATLAB Inference Bridge**: A fallback adapter allowing MATLAB to invoke the Python inference CLI when ONNX native execution is unavailable.

---

## Architecture Flow

```
[APTOS Dataset]
      ↓
[PyTorch Training Pipeline: python/training/train.py]
      ↓
[Best Checkpoint: models/aptos_efficientnet/best_model.pt]
      ↓
[ONNX Exporter: python/deployment/export_model.py]
      ↓
┌───────────────────────────────────────────────┐
│ Export Artifacts                              │
│ • models/aptos_efficientnet/best_model.onnx   │
│ • core/grading/runONNXModel.m (Auto-generated)│
│ • models/aptos_efficientnet/metadata.json     │
│ • models/aptos_efficientnet/metrics.json      │
└───────────────────────────────────────────────┘
      ↓
[MATLAB gradeDR.m]
      ├── Option A: importNetworkFromONNX ('best_model.onnx')
      ├── Option B: runPythonInference.m (CLI bridge)
      └── Option C: Rule-based Heuristic (Fallback / Demo Mode)
```

---

## 1. Exporting the Model to ONNX

Run the export script after model training:

```bash
python python/deployment/export_model.py
```

This performs:
- Loading of `models/aptos_efficientnet/best_model.pt`
- Forward pass validation with dummy input
- ONNX export using Opset 17 with dynamic batch axes
- Direct numerical parity verification between PyTorch and ONNX Runtime (`max_diff < 1e-4`)
- Auto-generation of MATLAB wrapper `core/grading/runONNXModel.m`

---

## 2. MATLAB Loading and Inference

### Native ONNX (Preferred)

In MATLAB R2023b or later with Deep Learning Toolbox:

```matlab
% Import ONNX network
onnxPath = fullfile('models', 'aptos_efficientnet', 'best_model.onnx');
net = importNetworkFromONNX(onnxPath);

% Run inference via RetinaGuard wrapper
cfg = RGConfig();
img = imread('sample_fundus.jpg');
result = runONNXModel(img, cfg);
```

### Python Bridge (Fallback)

If Deep Learning Toolbox Converter for ONNX Model is not installed in MATLAB:

```matlab
result = runPythonInference(img, cfg);
```

`runPythonInference.m` sends the image to `python python/inference/predict.py --json` and decodes the structured response directly.

---

## 3. Standalone Python Inference & Validation

You can run standalone inference without MATLAB:

```bash
# Human-readable output
python python/inference/predict.py demo/sample_images/demo_case1_grade0.png

# Pure JSON output with Grad-CAM heatmap generation
python python/inference/predict.py demo/sample_images/demo_case3_grade2.png --json --gradcam

# Run full automated system verification (18/18 tests)
python python/test_end_to_end_sih.py
```

---

## 4. Environment Requirements

### Python
- Python 3.10+
- PyTorch 2.0+
- torchvision
- onnx 1.14+
- onnxruntime 1.15+

### MATLAB
- MATLAB R2022b or later
- Deep Learning Toolbox (recommended for native ONNX import)
- Image Processing Toolbox
