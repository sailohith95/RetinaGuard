# RetinaGuard — AI-Assisted Diabetic Retinopathy Screening System

> **SIH 2026 Problem Statement** | AI-Assisted Retinal Image Analysis Workstation  
> *Edge-deployable screening workstation integrating deep learning classification, computer vision lesion candidates, and explainable AI.*

---

## 1. Project Overview

**RetinaGuard** is an end-to-end clinical screening system designed to assist healthcare providers in the early detection and severity grading of Diabetic Retinopathy (DR) from color fundus photographs.

It combines an automated **Image Quality Assessment Gate**, pre-processing with green-channel CLAHE contrast enhancement, classical computer vision for anatomical landmarks (Optic Disc, Fovea, Retinal Vessels) and candidate lesion detection (Microaneurysms, Hard Exudates, Hemorrhages), with a locked **EXP-001** deep learning model (`EfficientNet-B0`) providing 5-class International Clinical Diabetic Retinopathy (ICDR) disease severity grading and true **Grad-CAM** saliency maps.

### Screening Pipeline Workflow

```
Retinal Fundus Photograph
           ↓
[Quality Assessment Gate] ── (Laplacian Sharpness, Illumination, Contrast, FOV)
           │
           ├─ UNGRADABLE (< 40 Score) ──→ [Clinical Recapture Guidance] (Grading Halted)
           ↓ GRADABLE / BORDERLINE
[Contrast Enhancement] ───────→ [Green-Channel CLAHE (2.2 Clip, 8×8 Grid)]
           ↓
[Anatomical Structures] ──────→ [Optic Disc (OD), Fovea Projection, Vessel Density %]
           ↓
[Lesion Candidate Analysis] ──→ [Microaneurysms, Disc-Masked Exudates, Hemorrhages, NV Risk]
           ↓
[Production Model EXP-001] ───→ [ONNX EfficientNet-B0 (5-Class Softmax Probabilities)]
           ↓
[Explainable AI] ─────────────→ [True Grad-CAM Heatmap Activation Overlay]
           ↓
[Decision Support & Report] ──→ [Referral Triage (Grade ≥ 2) + Print-Ready HTML Report]
```

---

## 2. Production Model Performance (EXP-001)

The production model checkpoint is locked to **EXP-001** (`models/aptos_efficientnet/best_model.onnx` and `best_model.pt`). All metrics are genuine measured values validated on the held-out APTOS stratified validation set ($N = 733$ images):

| Metric | Measured Validation Value |
| :--- | :--- |
| **Quadratic Weighted Kappa (QWK)** | **0.7342** |
| **Overall Accuracy** | **69.85%** |
| **Macro F1-Score** | **0.4981** |
| **Referable DR Sensitivity (Grade ≥ 2)** | **78.86%** |
| **Referable DR Specificity (Grade < 2)** | **92.64%** |
| **ONNX vs PyTorch Numerical Parity** | **Max diff < 4.05 × 10⁻⁶** |

### Per-Class Performance Breakdown

| ICDR Severity Grade | Class Description | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: |
| **Grade 0** | No Diabetic Retinopathy | 0.812 | 0.941 | 0.872 |
| **Grade 1** | Mild Non-Proliferative DR | 0.443 | 0.395 | 0.418 |
| **Grade 2** | Moderate Non-Proliferative DR | 0.684 | 0.582 | 0.629 |
| **Grade 3** | Severe Non-Proliferative DR | 0.286 | 0.436 | 0.345 |
| **Grade 4** | Proliferative Diabetic Retinopathy | 0.310 | 0.153 | 0.198 |

---

## 3. Technology Stack

- **Deep Learning Core**: PyTorch 2.x & Torchvision (Architecture: EfficientNet-B0, Focal Loss $\gamma = 2.0$, Balanced Class Weighting).
- **Deployment & Inference Engine**: ONNX Runtime (Opset 17, CPU Execution Provider, Dynamic Batch Axes).
- **Computer Vision & Morphometry**: OpenCV (Headless), NumPy, SciPy, Pillow.
- **Web Application Backend**: FastAPI 0.111.0, Uvicorn, Jinja2, Python-Multipart.
- **Web Frontend Workstation**: Native semantic HTML5, CSS3, ES6 JavaScript (Zero external CDN dependencies; 100% offline).
- **MATLAB Desktop Integration**: Native `importNetworkFromONNX` and MATLAB App Designer workstation (`app/RetinaGuardApp.m`).

---

## 4. Quick Start (Local Execution)

### Single Launch Command
From the project root directory:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python webapp/run.py
```

The workstation will start at `http://localhost:8000` (auto-detecting alternative ports if `8000` is occupied) and open in your default browser.

---

## 5. Public Web Deployment (Render / Cloud)

RetinaGuard is ready for cloud deployment via [Render](https://render.com) using the included `render.yaml` blueprint:

- **Runtime**: Python 3.11 (`.python-version` configured to `3.11.9`)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Endpoint**: `/health`

*(See [DEPLOYMENT.md](file:///c:/Users/T%20SAILOHITH/Desktop/SIH/RetinaGuard/DEPLOYMENT.md) for step-by-step cloud deployment instructions).*

---

## 6. Project Directory Structure

```
RetinaGuard/
├── app/
│   └── RetinaGuardApp.m          # MATLAB Desktop App Designer UI
├── core/                         # MATLAB core algorithms
│   ├── enhancement/              # CLAHE and border cropping
│   ├── grading/                  # ONNX model runner & Python CLI bridge
│   ├── lesions/                  # Morphological lesion detection
│   ├── quality/                  # Image quality assessment
│   ├── reporting/                # HTML clinical report generator
│   ├── screening/                # MATLAB screening orchestrator
│   └── structures/               # Optic disc, fovea, vessel density
├── models/
│   └── aptos_efficientnet/       # Production Model EXP-001
│       ├── best_model.onnx       # Production ONNX model graph
│       ├── best_model.onnx.data  # ONNX external tensor weights (16 MB)
│       ├── best_model.pt         # PyTorch checkpoint (16.3 MB)
│       ├── metadata.json         # Training configuration metadata
│       └── metrics.json          # Validated evaluation metrics
├── python/
│   ├── config/config.yaml        # Unified project hyperparameters & paths
│   ├── explainability/gradcam.py # True Grad-CAM saliency implementation
│   ├── inference/predict.py      # Standalone Python inference API
│   ├── models/efficientnet_dr.py # PyTorch EfficientNet-B0 architecture
│   ├── preprocessing/            # Retinal preprocessing & CLAHE
│   └── test_end_to_end_sih.py    # 18-point comprehensive test suite
├── webapp/                       # Localhost & Cloud Web Application
│   ├── main.py                   # FastAPI REST endpoints & route handlers
│   ├── run.py                    # Single-command launcher (HOST 0.0.0.0 / $PORT)
│   ├── requirements.txt          # Webapp dependencies
│   ├── services/                 # Modular business logic services
│   ├── static/                   # CSS and JavaScript workstation controllers
│   ├── templates/index.html      # Workstation responsive UI
│   ├── test_webapp.py            # Webapp unit test suite (15/15 tests)
│   └── test_live_server.py       # Live HTTP integration test suite
├── .gitignore                    # Strict Git exclusions (caches, datasets, secrets)
├── .python-version               # Python 3.11.9 pin
├── DEPLOYMENT.md                 # Step-by-step deployment guide
├── render.yaml                   # Infrastructure-as-code for Render deployment
├── requirements.txt              # Root production requirements
└── runAllTests.m                 # 15-point MATLAB test suite
```

---

## 7. Verification & Automated Testing

The system has passed automated test suites across all layers:

```bash
# Web Application Unit Tests (15/15 PASS)
python webapp/test_webapp.py

# Live Server Integration Tests (ALL PASS)
python webapp/test_live_server.py

# Comprehensive Python End-to-End Suite (18/18 PASS)
python python/test_end_to_end_sih.py
```

---

## 8. Limitations & Clinical Constraints

1. **Candidate Lesion Heuristics**: Microaneurysms, hard exudates, and hemorrhages are detected using classical mathematical morphology and edge filters. They represent *candidate detections* to assist physician review, not manual specialist segmentations.
2. **Field of View**: Validated primarily on standard 45-degree macula-centered or disc-centered fundus photographs. Performance on ultra-widefield (UWF) scanning laser ophthalmoscopy is not calibrated.
3. **Class Imbalance in Proliferative DR**: While minority-class performance was significantly enhanced via Focal Loss, severe NPDR (Grade 3) and Proliferative DR (Grade 4) remain low-prevalence classes in public datasets.

---

## 9. Important Clinical Disclaimer

> **INVESTIGATIONAL / SCREENING DECISION SUPPORT PROTOTYPE ONLY**  
> RetinaGuard is an artificial intelligence decision-support prototype developed for research and educational purposes. **It does not provide an autonomous medical diagnosis and has not undergone formal regulatory clinical trials (FDA, CE, or CDSCO).** All automated assessments, severity classifications, and triage recommendations must be reviewed and confirmed by a licensed ophthalmologist or qualified healthcare professional before making patient management decisions.
