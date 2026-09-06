# RetinaGuard — Localhost Web Application

> **AI-Assisted Diabetic Retinopathy Screening Workstation**  
> *Companion browser interface for clinical screening, edge deployment, and live demonstrations.*

---

## 1. Overview

The **RetinaGuard Localhost Web Application** is a browser-accessible, offline-first clinical workstation providing full access to the RetinaGuard screening pipeline without requiring MATLAB Desktop.

It runs locally on your machine via a lightweight Python/FastAPI backend and native modern HTML5/CSS3/JavaScript frontend. It uses the exact production **EXP-001** model (`models/aptos_efficientnet/best_model.pt`) and shared algorithmic modules.

### Key Capabilities
- **Single-Command Launch**: Automated server initialization and browser launch.
- **Production Model**: EXP-001 EfficientNet-B0 ($\text{QWK}=0.7342$, $\text{Accuracy}=69.85\%$, $\text{Referable Sensitivity}=78.86\%$, $\text{Referable Specificity}=92.64\%$).
- **Ungradable Safety Gate**: Strictly halts AI grading when image quality falls below diagnostic thresholds, preventing false negative diagnoses.
- **Multimodal Visualization**: Interactive layer switching across Original, Contrast-Enhanced (CLAHE), Retinal Vessels, Candidate Lesions, and True Grad-CAM Heatmaps.
- **Clinical Reporting**: Automated, printable HTML clinical report with patient metadata, quantitative findings, and audit disclaimers.
- **100% Offline & Private**: Zero external CDN calls, telemetry, or cloud dependencies. Patient data never leaves localhost.

---

## 2. Quick Start

### The ONE Launch Command

From the project root directory (`c:\Users\T SAILOHITH\Desktop\SIH\RetinaGuard`):

```bash
python webapp/run.py
```

This command will:
1. Initialize the FastAPI backend on port `8000` (or the next available port if `8000` is busy).
2. Automatically launch your default web browser to:
   ```
   http://localhost:8000
   ```
3. Prepare all pre-trained weights and demo cases for instant screening.

---

## 3. System Requirements & Dependencies

The web application runs on standard Python 3.9+ environments. All dependencies are minimal and already part of the project environment:

- `fastapi` & `uvicorn` (ASGI web framework & server)
- `jinja2` & `python-multipart` (template rendering & file upload streaming)
- `torch` & `torchvision` (EXP-001 deep learning inference & Grad-CAM)
- `opencv-python` & `numpy` (retinal computer vision & morphological analysis)
- `scipy` & `matplotlib` (colormap generation & morphological kernels)

*(See `webapp/requirements.txt` for exact pins if installing in a clean virtual environment).*

---

## 4. Workstation User Interface Guide

### A. Image Ingestion
- **Custom Image Upload**: Click or drag-and-drop any standard fundus photograph (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`).
- **Built-In Demo Cases**: Select any of the 6 pre-loaded reference cases from the dropdown:
  - **Case 1**: Normal Retina (Grade 0 — No DR)
  - **Case 2**: Mild NPDR Reference (Grade 1)
  - **Case 3**: Moderate NPDR Reference (Grade 2)
  - **Case 4**: Severe NPDR Reference (Grade 3)
  - **Case 5**: Proliferative DR Reference (Grade 4)
  - **Case 6**: Poor Quality / Blurred (Ungradable Safety Gate Trigger)

### B. Interactive Retina Viewport
Switch between analytical image layers with the viewport toggle buttons:
- **Original**: Unaltered input fundus photograph.
- **Enhanced (CLAHE)**: Green-channel Contrast Limited Adaptive Histogram Equalization with circular FOV mask.
- **Vessel Map**: Segmented retinal vascular tree and vascular density metric.
- **Lesions Overlay**: Candidate microaneurysms (red), hard exudates (yellow, optic disc excluded), hemorrhages (cyan), and neovascularization indicator.
- **Grad-CAM**: Real gradient-weighted class activation mapping highlighting the exact retinal regions driving the neural network's grade.

### C. Clinical Decision Support Panel
- **Image Quality Assessment**: Real-time sharpness (Laplacian variance), illumination, contrast, and composite Quality Score ($0\text{--}100$).
- **ICDR Grade & Confidence**: Predicted grade ($0\text{ to }4$), diagnostic category, softmax confidence percentage, and full class probability breakdown.
- **Referral Triage**: Clear visual badge indicating **NON-REFERABLE** (Routine 12-month follow-up) or **REFERABLE DR** (Prompt/urgent ophthalmologist referral required).
- **Candidate Lesion Summary**: Quantitative counts of candidate microaneurysms, exudates, and hemorrhages.
- **Download Clinical Report**: Generates and downloads a self-contained HTML medical report.

---

## 5. Verification & Testing

RetinaGuard includes dedicated automated test suites for both unit and live server verification:

### Run Web Application Unit Tests (15 Tests)
```bash
python webapp/test_webapp.py
```
*Validates API endpoints, upload handling, quality scoring, safety gate, EXP-001 inference parity, Grad-CAM generation, and report synthesis.*

### Run Live Server Integration Test
With the server running:
```bash
python webapp/test_live_server.py
```
*Executes live HTTP requests against `http://127.0.0.1:8000`, verifying health status, HTML delivery, demo loading, Grade 3 analysis, and Case 6 safety rejection.*

### Run Full System End-to-End Suite (18 Tests)
```bash
python python/test_end_to_end_sih.py
```
*Ensures 100% backward compatibility and zero regressions across the core computer vision and PyTorch/ONNX pipelines.*

---

## 6. Architecture & File Structure

```
webapp/
├── __init__.py               # Starlette/FastAPI runtime compatibility shim
├── main.py                   # FastAPI REST API endpoints & route handlers
├── run.py                    # Production single-command launcher with port fallback
├── requirements.txt          # Python dependencies
├── test_webapp.py            # Unit test suite (15 tests, TestClient)
├── test_live_server.py       # Live HTTP integration test suite
├── README.md                 # This documentation
├── services/                 # Modular business logic services
│   ├── __init__.py
│   ├── quality_service.py    # Sharpness, illumination, FOV, quality score
│   ├── enhancement_service.py# Circular crop, green-channel CLAHE
│   ├── structure_service.py  # Optic disc, fovea, vessel segmentation & density
│   ├── lesion_service.py     # Candidate MAs, exudates, HEs, NV risk
│   ├── grading_service.py    # EXP-001 PyTorch inference & Grad-CAM
│   ├── report_service.py     # HTML report compiler
│   └── screening_service.py  # Master coordinator & safety gate
├── static/                   # Static assets (100% offline)
│   ├── css/
│   │   └── style.css         # Clinical workstation UI styling
│   └── js/
│       └── app.js            # Viewport controller, API client & interactions
└── templates/
    └── index.html            # Workstation HTML5 interface
```

---

## 7. Medical Device Disclaimer

> **INVESTIGATIONAL / DECISION SUPPORT USE ONLY**  
> RetinaGuard is an AI-assisted decision support screening prototype designed for research and demonstration purposes. It does **not** provide a definitive medical diagnosis. All screening evaluations, candidate lesion detections, and triage recommendations must be independently reviewed by a qualified ophthalmologist or licensed healthcare professional before making clinical decisions.
