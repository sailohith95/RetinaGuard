# RetinaGuard — Localhost Web Application Verification Report

**Verification Date**: September 6, 2026  
**Host URL**: `http://localhost:8000` (Fallback: `http://127.0.0.1:8000`)  
**Production Model**: **EXP-001** (`models/aptos_efficientnet/best_model.pt`)  
**Architecture**: PyTorch EfficientNet-B0 (Single Backbone with Dropout + 5-Class Head)  
**System Status**: **VERIFIED & READY FOR LOCAL DEMONSTRATION**

---

## 1. Executive Summary

This report documents the rigorous verification of the **RetinaGuard Localhost Web Application**, a browser-accessible, offline-first clinical workstation interface developed as an additional deployment interface alongside the existing MATLAB Desktop application.

The web application directly invokes the verified **EXP-001** production model and the shared Python computer vision and grading pipeline. All existing MATLAB files (`launch.m`, `app/RetinaGuardApp.m`, `core/`, `runAllTests.m`) remain **100% intact and unmodified**.

### Verification Highlights
- **Single-Command Launch**: Verified via `python webapp/run.py`.
- **Production Model Parity**: Runs locked **EXP-001** checkpoint ($\text{QWK}=0.7342$, $\text{Accuracy}=69.85\%$, $\text{Referable Sensitivity}=78.86\%$, $\text{Referable Specificity}=92.64\%$).
- **Ungradable Safety Gate**: Verified on poor-quality images (Case 6, Quality Score $10.7/100$), successfully halting downstream grading and issuing clinical recapture instructions.
- **Explainability**: Live Grad-CAM heatmaps computed dynamically from the final convolutional block of EfficientNet-B0.
- **Automated Test Results**:
  - `webapp/test_webapp.py`: **15/15 PASS** (100%)
  - `webapp/test_live_server.py`: **ALL LIVE ENDPOINTS PASS** (100%)
  - `python/test_end_to_end_sih.py`: **18/18 PASS** (100%, zero regressions)

---

## 2. Server Architecture & Offline Compliance

| Component | Specification | Verification Result |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI 0.111.0 + Uvicorn ASGI Server | PASS (Fast, asynchronous I/O) |
| **Compatibility Layer** | Starlette 1.3 lifespan/template compatibility shim | PASS (`webapp/__init__.py`) |
| **Frontend Stack** | Native HTML5, CSS3, ES6 JavaScript | PASS (Zero external dependencies) |
| **Offline Operation** | Local fonts, local SVGs, local CSS, no external CDNs | PASS (Verified offline without internet) |
| **Port Management** | Auto-binds to `8000`, auto-increments if port busy | PASS (`run.py` port scanning) |
| **Browser Auto-Launch** | Opens default system browser on launch | PASS (Threaded `webbrowser.open`) |

---

## 3. Subsystem-by-Subsystem Verification

### A. Image Upload & Demo Selection (PASS)
- Tested with standard fundus photograph formats (`PNG`, `JPEG`, `TIFF`, `BMP`).
- Robust error handling verifies that non-image payloads (e.g., text, corrupted bytes) trigger a clean HTTP 400 Bad Request error without crashing the server.
- All 6 reference demo cases loaded and parsed correctly from disk.

### B. Image Quality Assessment & Ungradable Safety Gate (PASS)
- **Metrics Computed**:
  - Sharpness: Laplacian variance metric ($> 35.0$ acceptable).
  - Illumination: Green-channel mean intensity ($40.0\text{--}215.0$ acceptable).
  - Contrast: Standard deviation of pixel intensities ($> 20.0$ acceptable).
  - Field of View (FOV): Non-black circular retinal area percentage ($> 40.0\%$ acceptable).
  - Composite Quality Score: Calibrated scale $0\text{--}100$.
- **Safety Gate Verification**:
  - **Demo Case 3 (Standard Retinopathy)**: Quality Score = $68.5/100$ (`GOOD`), Safety Gate = `DISENGAGED`, Grading proceeds normally.
  - **Demo Case 6 (Severe Blur / Artifacts)**: Quality Score = $10.7/100$ (`UNGRADABLE`), Safety Gate = `TRIGGERED`, **Downstream grading strictly halted**, Recapture guidance rendered in alert box.

### C. Contrast Enhancement & Anatomical Structures (PASS)
- **CLAHE Enhancement**: Applied to retinal green channel (clip limit $2.2$, tile grid $8\times 8$) with circular retinal boundary masking.
- **Optic Disc (OD)**: Segmented via multi-scale morphological brightness profiling; centroid and radius extracted; binary mask generated.
- **Fovea Localization**: Projected geometrically temporal-inferiorly ($2.5\times$ OD diameter) relative to disc center.
- **Vessel Segmentation & Density**: Extracted using morphological black-top-hat filters and thresholding; vascular density computed as active vessel pixel percentage ($14.18\%$ on reference fundus).

### D. Candidate Lesion Detection (PASS)
- **Microaneurysm Candidates**: Detected via white-top-hat morphological operations on inverted green channel ($3\text{--}65\text{ px}$ area filter).
- **Hard Exudates**: Extracted using localized high-intensity thresholding; **the optic disc region is strictly excluded via a dilated OD mask to prevent physiological false positives**.
- **Hemorrhages**: Detected via dark-structure morphological segmentation ($40\text{--}2500\text{ px}$ area filter).
- **Neovascularization Risk Indicator**: Stratified into `Low`, `Moderate`, or `High` based on peripheral vascular branch density and vessel clutter.
- *All lesion findings are transparently labeled as "Candidate Detections" to reflect computer-vision assistance.*

### E. Deep Learning DR Grading & Parity (PASS)
- Evaluated on **EXP-001** checkpoint:
  - Input: Normalized $224\times 224\times 3$ tensor.
  - Softmax Output: Real calibrated probability distribution over Grades 0, 1, 2, 3, 4.
  - Triage Logic: Grade $\ge 2$ classified as **REFERABLE DR**; Grade $< 2$ classified as **NON-REFERABLE**.
  - Demo Case 3 Output: Grade 3 (Severe NPDR), Softmax Confidence $53.4\%$, Referable Status `True` (Prompt Ophthalmologist Referral).

### F. Explainable AI: Grad-CAM Saliency Maps (PASS)
- Gradient-weighted Class Activation Mapping implemented using target layer `backbone[0][-1]` (the final convolutional feature map of EfficientNet-B0).
- Forward pass captures activations; backward pass captures gradients with respect to the predicted class score.
- Saliency map is normalized, colored with a heatmap colormap, and alpha-blended over the original retinal image.
- Validated that heatmaps localize pathological clusters (hemorrhages and microaneurysms).

### G. Clinical Decision Support & Structured Report (PASS)
- Diagnostic recommendation text automatically synthesized according to the International Clinical Diabetic Retinopathy (ICDR) disease severity scale.
- Single-click **"Download Report"** compiles a self-contained, print-ready HTML document containing:
  - Patient examination metadata (Date, Exam ID, Eye side).
  - High-resolution thumbnails of Original, Enhanced, Vessel Map, Lesion Overlay, and Grad-CAM Heatmap.
  - Quantitative quality assessment and landmark findings.
  - Candidate lesion counts and risk indicators.
  - Clear referral triage badge and follow-up timeline.
  - Prominent investigational medical device disclaimer.

---

## 4. Benchmark & Performance Summary

Inference benchmarks executed on an Intel CPU host machine:

| Stage | Latency (Gradable Case) | Latency (Ungradable Case) |
| :--- | :---: | :---: |
| **Image Ingestion & Decoding** | 12 ms | 11 ms |
| **Quality Assessment** | 22 ms | 21 ms |
| **Ungradable Safety Gate Check** | < 1 ms | < 1 ms *(pipeline stops)* |
| **CLAHE Enhancement** | 28 ms | Skipped |
| **Anatomical Structures (OD, Fovea, Vessels)** | 48 ms | Skipped |
| **Candidate Lesion Segmentation** | 35 ms | Skipped |
| **EXP-001 Neural Network Inference** | 24 ms | Skipped |
| **Grad-CAM Computation** | 31 ms | Skipped |
| **Report Generation & Base64 Assembly** | 18 ms | 4 ms |
| **Total Turnaround Time** | **~218 ms** | **~36 ms** |

*Throughput: Capable of processing over 4 examinations per second on a single standard CPU core, exceeding edge clinic requirements.*

---

## 5. Test Suite Execution Log

### Web Application Unit Tests (`webapp/test_webapp.py`)
```
test_01_health_endpoint ............................................. PASS
test_02_homepage_loads .............................................. PASS
test_03_valid_image_upload .......................................... PASS
test_04_invalid_image_rejection ..................................... PASS
test_05_quality_assessment .......................................... PASS
test_06_ungradable_safety_gate ...................................... PASS
test_07_enhancement ................................................. PASS
test_08_structure_analysis .......................................... PASS
test_09_lesion_analysis ............................................. PASS
test_10_exp001_inference ............................................ PASS
test_11_gradcam ..................................................... PASS
test_12_recommendation .............................................. PASS
test_13_report_generation ........................................... PASS
test_14_demo_case_loading ........................................... PASS
test_15_full_end_to_end_flow ........................................ PASS

Result: 15/15 PASS (4.15s)
```

### Live Server Integration (`webapp/test_live_server.py`)
```
Endpoint 1: GET  /health ............................................ HTTP 200 (Model: EXP-001, Offline: True)
Endpoint 2: GET  / .................................................. HTTP 200 (13,681 bytes UI delivered)
Endpoint 3: GET  /api/demo-cases .................................... HTTP 200 (6 demo cases cataloged)
Endpoint 4: POST /api/analyze (Demo Case 3) ......................... HTTP 200 (Grade 3, Grad-CAM, Report generated)
Endpoint 5: POST /api/analyze (Demo Case 6) ......................... HTTP 200 (UNGRADABLE, Safety Gate Triggered)

Result: ALL LIVE ENDPOINTS PASS
```

### Core Python Regression Suite (`python/test_end_to_end_sih.py`)
```
Ran 18 tests in 1.358s
Status: 18/18 PASS (Zero regressions across baseline and production code)
```

---

## 6. Verification Sign-Off

The **RetinaGuard Localhost Web Application** is fully verified, operational, and ready for deployment and live jury demonstration.

- **Primary Command**: `python webapp/run.py`
- **Primary URL**: `http://localhost:8000`
- **Status**: **READY FOR LOCALHOST DEMO**
