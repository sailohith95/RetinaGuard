# RetinaGuard — AI-Assisted Diabetic Retinopathy Screening System

> **SIH 2026 Problem Statement** | AI-Assisted Retinal Image Analysis Workstation  
> *Edge-deployable screening workstation integrating deep learning classification, computer vision lesion candidates, and explainable AI.*

---

## 1. Project Overview

**RetinaGuard** is an offline-capable, clinical-grade decision support system engineered for early detection, anatomical assessment, and severity grading of Diabetic Retinopathy (DR) from digital color fundus photographs. 

Designed for deployment in primary healthcare centers, rural clinics, and tele-ophthalmology screening camps, RetinaGuard bridges the gap between patient presentation and specialist care by providing real-time, explainable triage without requiring continuous internet connectivity.

The system features two operational interfaces:
1. **Localhost Web Workstation**: A modern, responsive browser interface built on FastAPI and native web standards.
2. **MATLAB Desktop Workstation**: A standalone clinical App Designer interface supporting native MATLAB workflows.

---

## 2. SIH Problem Statement Mapping

| SIH Requirement | RetinaGuard Implementation | Verification |
| :--- | :--- | :--- |
| **Automated Image Quality Assessment** | Multi-metric gate analyzing Laplacian sharpness, mean illumination, Michelson contrast, and circular Field-of-View (FOV). | Blocks ungradable images (< 40 score) to prevent misdiagnosis. |
| **Image Preprocessing & Enhancement** | Green-channel extraction, circular FOV border masking, and Contrast Limited Adaptive Histogram Equalization (CLAHE). | Calibrated 2.2 clip limit on 8×8 contextual grid. |
| **Anatomical Landmark Detection** | Morphological peak localization for Optic Disc (OD), temporal geometric projection for Fovea, and vessel segmentation for vascular density %. | Fully automated with disc exclusion mask generation. |
| **Lesion Candidate Detection** | Top-hat morphological filtering for microaneurysms, luminance thresholding with disc masking for hard exudates, and dark-lesion segmentation for hemorrhages. | Candidate detection counts and Neovascularization Risk Indicator. |
| **Multi-Class DR Severity Grading** | 5-class International Clinical Diabetic Retinopathy (ICDR) grading (Grades 0 to 4) using locked production model **EXP-001** (`EfficientNet-B0`). | $\text{QWK} = 0.7342$, $\text{Accuracy} = 69.85\%$, $\text{Macro F1} = 0.4981$. |
| **Clinical Decision Support & Triage** | Automated binary referral classification (Grade $\ge 2$ = Referable DR), urgency guidelines, and follow-up intervals. | $\text{Sensitivity} = 78.86\%$, $\text{Specificity} = 92.64\%$. |
| **Explainable AI (XAI)** | Full gradient-weighted class activation mapping (Grad-CAM) overlaid directly onto the retinal image. | PyTorch backward pass targeting final convolutional feature maps (`backbone.0.8`). |
| **Auditable Clinical Reporting** | Automated, print-ready, self-contained HTML medical screening report with patient metadata, image metrics, and legal disclaimers. | Exportable and downloadable directly from workstation UI. |

---

## 3. Architecture

RetinaGuard employs a 7-stage deterministic screening pipeline:

```
                  Raw Color Fundus Photograph
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   STAGE 1: Quality Assessment Gate   │
            │  (Sharpness, Illumination, Contrast) │
            └──────────────────┬───────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
      [Score < 40]                        [Score ≥ 40]
             │                                   │
             ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│     SAFETY GATE HALT    │         │  STAGE 2: Enhancement   │
│ Downstream grading      │         │   Green-channel CLAHE   │
│ stopped; clinical       │         └────────────┬────────────┘
│ recapture requested     │                      │
└─────────────────────────┘                      ▼
                                    ┌─────────────────────────┐
                                    │   STAGE 3: Landmarks    │
                                    │ Optic Disc, Fovea,      │
                                    │ Vessel Density %        │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │    STAGE 4: Lesions     │
                                    │ Microaneurysms,         │
                                    │ Disc-Masked Exudates,   │
                                    │ Hemorrhages, NV Risk    │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │ STAGE 5: EXP-001 ONNX   │
                                    │ 5-Class ICDR Grading    │
                                    │ Softmax Probabilities   │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │  STAGE 6: Real Grad-CAM │
                                    │ True Saliency Heatmap   │
                                    │ Visual Explanation      │
                                    └────────────┬────────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────────┐
                                    │ STAGE 7: Decision & PDF │
                                    │ Triage Recommendation   │
                                    │ Print-Ready HTML Report │
                                    └─────────────────────────┘
```

---

## 4. Features

- **Double-Workstation Architecture**: Choose between the zero-install web browser interface or the native MATLAB desktop app.
- **True Grad-CAM Explainability**: Highlights the spatial regions driving the model's prediction without using surrogate or fake heatmaps.
- **Strict Patient Safety Gate**: Immediately aborts classification on severely blurred, underexposed, or corrupted images to eliminate false reassurances.
- **Multi-Layer Interactive Viewport**: Instantly toggle between Original, CLAHE Enhanced, Vessel Map, Lesion Candidates, and Grad-CAM Heatmap views.
- **Integrated Report Generator**: Produces auditable, printable medical screening reports in HTML format.
- **100% Offline Capable**: Requires no internet access or external cloud services; ideal for isolated rural clinics.

---

## 5. Requirements

### Hardware Requirements
- **Processor**: Dual-Core CPU (Intel Core i3 / AMD Ryzen 3 or higher)
- **RAM**: Minimum 4 GB RAM (8 GB recommended)
- **Disk Space**: ~2 GB free disk space for models, dependencies, and sample images

### Software Requirements
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python**: Python 3.9, 3.10, or 3.11
- **MATLAB (Optional)**: MATLAB R2021b or newer with Deep Learning Toolbox and Image Processing Toolbox (only required for running the MATLAB app).

---

## 6. Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/sailohith95/RetinaGuard.git
cd RetinaGuard
```

### Python Environment Setup

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Install the production dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. Local Startup

### Running the Web Application (Recommended)

From the project root directory, run:

```bash
python webapp/run.py
```

- The launcher will automatically bind to `http://127.0.0.1:8000` (or the next available port).
- Your default web browser will open to the clinical workstation interface automatically.
- To access from other devices on your local Wi-Fi, navigate to `http://<your-computer-ip>:8000`.

### Running the MATLAB Application

Open MATLAB and navigate to the `RetinaGuard` directory, then run:

```matlab
launch
```

This will initialize the project paths, verify model weights, and launch the MATLAB App Designer workstation.

---

## 8. Demo Instructions

The workstation includes **6 curated authentic reference cases** covering the entire clinical spectrum:

| Case ID | Scenario | True ICDR Grade | Expected Behavior |
| :--- | :--- | :---: | :--- |
| **Case 1** | Normal Healthy Retina | Grade 0 (No DR) | Passes quality gate; predicts Grade 0; Non-Referable; routine 12-month follow-up. |
| **Case 2** | Mild NPDR Reference | Grade 1 (Mild NPDR) | Detects early microaneurysms; non-referable triage. |
| **Case 3** | Moderate NPDR Reference | Grade 2 (Moderate NPDR) | Detects hard exudates & microaneurysms; triggers Referable DR badge (3–6 month referral). |
| **Case 4** | Severe NPDR Reference | Grade 3 (Severe NPDR) | Multiple intra-retinal hemorrhages; triggers Urgent Referable DR (1 month). |
| **Case 5** | Proliferative DR Reference | Grade 4 (PDR) | High lesion density and neovascularization risk; triggers immediate specialist referral. |
| **Case 6** | Poor Quality / Motion Blur | Ungradable (-1) | **Safety Gate triggers**: Downstream grading halted, recapture guidance displayed. |

### How to Run a Demo
1. In the web workstation, locate the **Curated Reference Scenarios** dropdown.
2. Select any case (e.g., *Case 3 — Moderate NPDR Reference*).
3. Click **Load Case** to display the retinal photograph.
4. Click **Analyze Image ▶**. The full multi-stage pipeline executes in under 1 second.
5. Use the layer toggle buttons (**Original**, **Enhanced**, **Vessels**, **Lesions**, **Heatmap**) to explore the findings.
6. Click **Download PDF / Clinical Report** to export the printable documentation.

---

## 9. Model Information

The primary screening model is locked to **EXP-001**:

- **Architecture**: `EfficientNet-B0` (Pre-trained on ImageNet, fine-tuned on fundus photography)
- **Loss Function**: Focal Loss ($\gamma = 2.0$) with inverse class-frequency weighting to address medical class imbalance
- **Input Resolution**: $224 \times 224$ pixels, standardized with retinal circle cropping and CLAHE
- **Inference Format**: Dual deployment:
  - **ONNX Runtime** (`models/aptos_efficientnet/best_model.onnx`): High-speed, low-memory CPU forward pass (< 0.15s)
  - **PyTorch** (`models/aptos_efficientnet/best_model.pt`): Full autograd engine for true Grad-CAM computation

### Verified Production Metrics

Validated on the held-out stratified APTOS validation set ($N = 733$ images):

| Metric | Validated Score |
| :--- | :--- |
| **Quadratic Weighted Kappa (QWK)** | **0.7342** |
| **Overall Accuracy** | **69.85%** |
| **Macro F1-Score** | **0.4981** |
| **Referable DR Sensitivity (Grade ≥ 2)** | **78.86%** |
| **Referable DR Specificity (Grade < 2)** | **92.64%** |
| **ONNX vs PyTorch Numerical Parity** | **Max absolute error < 4.05 × 10⁻⁶** |

---

## 10. Dataset Information

RetinaGuard was trained and evaluated on the **APTOS 2019 Blindness Detection** benchmark dataset:

- **Total Images**: 3,662 high-resolution color fundus photographs acquired across rural clinics in India.
- **Split Strategy**: 80% Stratified Training ($N = 2,929$), 20% Held-out Stratified Validation ($N = 733$).
- **Class Distribution**:
  - Grade 0 (No DR): 1,805 images (49.3%)
  - Grade 1 (Mild NPDR): 370 images (10.1%)
  - Grade 2 (Moderate NPDR): 999 images (27.3%)
  - Grade 3 (Severe NPDR): 193 images (5.3%)
  - Grade 4 (Proliferative DR): 295 images (8.0%)
- **Data Privacy Note**: Raw dataset images and competition archives are excluded from the repository via `.gitignore`. The repository includes 6 anonymized demo images strictly for validation and testing.

---

## 11. MATLAB Setup

If you wish to use the native MATLAB interface:

1. Open MATLAB (R2021b or later recommended).
2. Set your current folder to the cloned `RetinaGuard` directory.
3. Run the automated test suite:
   ```matlab
   runAllTests
   ```
4. Launch the desktop application:
   ```matlab
   launch
   ```

*Required MATLAB Toolboxes: Deep Learning Toolbox, Image Processing Toolbox, Computer Vision Toolbox.*

---

## 12. Python Setup

Verify your Python environment by running the environment audit:

```bash
python python/audit_pipeline.py
```

This verifies that:
- Python version is 3.9–3.11
- PyTorch and Torchvision can instantiate `EfficientNet-B0`
- ONNX Runtime loads `models/aptos_efficientnet/best_model.onnx`
- OpenCV is functioning in headless mode
- All demo sample images are present in `demo/sample_images/`

---

## 13. Testing

RetinaGuard includes comprehensive automated test suites covering all components:

### 1. Web Application Test Suite (15 Tests)
Validates web endpoints, quality scoring, safety gate, EXP-001 inference, Grad-CAM, and reporting:
```bash
python webapp/test_webapp.py
```

### 2. Live Server Integration Test
Validates live HTTP request/response flows against a running server:
```bash
# In terminal 1: start server
python webapp/run.py

# In terminal 2: run test
python webapp/test_live_server.py
```

### 3. Core SIH End-to-End Suite (18 Tests)
Validates image enhancement, anatomical landmarks, lesion detectors, PyTorch/ONNX parity, and latency benchmarks:
```bash
python python/test_end_to_end_sih.py
```

### 4. MATLAB Test Suite (15 Tests)
In MATLAB:
```matlab
runAllTests
```

---

## 14. Limitations

1. **Candidate Lesion Heuristics**: Microaneurysms, exudates, and hemorrhages are identified using mathematical morphology and edge filters. They serve as *candidate visual aids* for clinicians, not certified manual segmentations.
2. **Camera Calibration**: Calibrated primarily for standard 45° to 50° macula-centered fundus photographs. Performance on ultra-widefield (UWF) scanning laser systems has not been calibrated.
3. **Severe Pathology Rarity**: Due to the natural epidemiological distribution of diabetic retinopathy, Grades 3 and 4 represent smaller proportions of public datasets. Clinicians should exercise particular care with suspected advanced pathology.

---

## 15. AI-Assisted Screening Disclaimer

> **IMPORTANT CLINICAL & LEGAL NOTICE**  
> **RetinaGuard is an artificial intelligence research and decision-support prototype.**  
> It is designed to assist qualified healthcare professionals by triaging fundus images and highlighting suspicious visual findings. **It does NOT constitute an autonomous medical device, does not provide a definitive diagnosis, and has not been certified by the US FDA, European CE, or Indian CDSCO.**  
> All automated classifications, quality evaluations, and triage recommendations must be independently verified by a licensed ophthalmologist, optometrist, or trained medical practitioner before initiating or altering any patient clinical management plan.
