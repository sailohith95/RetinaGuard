# RetinaGuard Application Walkthrough

## 1. Starting the Application

To start the RetinaGuard Clinical Screening Workstation, launch MATLAB and execute:

```matlab
% In the MATLAB Command Window:
cd('C:\Users\T SAILOHITH\Desktop\SIH\RetinaGuard');
run('launch.m');
```

The startup routine checks MATLAB version, verifies the Image Processing Toolbox, initializes paths from `config/RGConfig.m`, detects the trained production model (`models/aptos_efficientnet/best_model.onnx` or `best_model.pt`), and constructs the multi-panel `RetinaGuardApp` interface.

Alternatively, for standalone headless verification via Python:
```bash
python python/test_end_to_end_sih.py
```

---

## 2. Main Screen

The application presents a unified clinical workstation divided into three functional zones:

1. **Top Header Bar & Navigation Sidebar:**
   - Displays "RetinaGuard — AI-Assisted DR Screening | v1.0.0".
   - Left navigation sidebar providing six primary views:
     - **Screening Workstation** (primary diagnostic view)
     - **Patients** (demographics & session roster)
     - **Examinations** (examination history)
     - **Reports** (generated HTML clinical summaries)
     - **Model / AI** (model metadata, QWK, validation scorecards)
     - **System** (configuration paths and toolbox diagnostics)

2. **Central Image Workspace:**
   - High-resolution interactive retinal viewport (`AxImage`).
   - Control bar containing:
     - **Upload Image**: Opens file browser for external clinical fundus scans (`.jpg`, `.png`, `.tif`).
     - **Demo Dropdown**: Selects from 6 built-in authentic screening cases.
     - **Load Demo**: Loads selected case image into the viewport.
     - **Analyze Image ▶**: Initiates the automated 5-stage screening pipeline.
     - **Layer Toggles**: `Original`, `Enhanced`, `Vessels`, `Lesions`, `Heatmap`.

3. **Clinical Results & Findings Panels:**
   - **Examination Info**: Patient ID, Exam ID, Timestamp, Eye (OD/OS), Examination Status.
   - **Image Quality Assessment**: Overall Score (0–100), Status badge (`GOOD`, `BORDERLINE`, `UNGRADABLE`), Focus, Illumination, FOV, and Contrast.
   - **Retinal Anatomical Landmarks**: Optic Disc localization, Fovea centralis coordinates, and Retinal Vessel Density (%).
   - **Lesion Candidate Analysis**: Microaneurysm candidates, Hard exudate candidates (disc-masked), Hemorrhage candidates, and Neovascularization Risk Indicator.
   - **DR Severity Result**: 5-class ICDR grade (0–4), confidence %, referable indicator, and clinical recommendation.

---

## 3. Loading an Image

Users can load fundus scans through two methods:
1. **Clinical File Ingestion:** Click **Upload Image**, choose a fundus photograph from the file system. Supported formats include JPG, PNG, and TIFF.
2. **Demo Case Selection:** Choose a scenario from the **Demo dropdown** (e.g. *Normal Retina — No DR*, *Moderate Non-Proliferative DR*, or *Poor Quality — Ungradable*), then click **Load Demo**. The authentic fundus scan loads into the central viewer immediately.

Upon loading, examination metadata (Patient ID `DEMO-P000X`, Exam ID `DEMO-E000X`, date/time) is automatically assigned.

---

## 4. Image Quality Assessment

Before deep learning inference is performed, the image passes through the automated quality gate:
- **Sharpness/Focus**: Quantified by Laplacian gradient variance ($\sigma^2_{lap}$).
- **Illumination**: Evaluates mean green-channel luminance to detect under- or over-exposure ($35 \le \mu \le 225$).
- **Contrast**: Evaluates standard deviation and dynamic range.
- **Overall Score**: Weighted score normalized from 0 to 100.
  - **GOOD ($\ge 60$)**: Full pipeline proceeds normally.
  - **BORDERLINE ($40\text{--}59$)**: Pipeline proceeds with an alert advising clinician verification.
  - **UNGRADABLE ($< 40$)**: Downstream grading is immediately halted to prevent diagnostic error.

---

## 5. Enhancement

Clicking the **Enhanced** toggle applies clinical contrast enhancement:
- **Border Cropping**: Strips non-retinal black camera borders.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Equalizes the green channel with a clip limit of 2.0 and an $8 \times 8$ tile grid.
- **Denoising**: Median and Gaussian filtering to reduce sensor noise while preserving fine vessel edges.
- **Comparison**: Clinicians can toggle between **Original** and **Enhanced** to inspect subtle micro-vascular pathology.

---

## 6. Retinal Structures

The structure analysis module isolates key anatomical landmarks:
- **Optic Disc**: Detected via peak luminance clustering. Outputs centroid $[c_y, c_x]$, radius $r_{od}$, and generates an explicit circular binary mask (`odMask`).
- **Fovea Centralis**: Positioned ~2.5 disc diameters temporal to the optic disc and refined by searching for the local macula luminance depression.
- **Retinal Vasculature**: Segmented using multi-scale Gaussian ridge filtering and black top-hat morphology; reports **Vessel Density (%)** (normal range: 5%–18% of retinal surface area).

---

## 7. Lesion Analysis

Quantitative computer-vision candidate detection assists the screening review:
- **Microaneurysm Candidates**: Morphological top-hat filter isolates small, circular, dark lesions ($3\text{--}65\text{ px}$).
- **Hard Exudate Candidates**: Identifies bright, lipid-rich deposits. The optic disc region is strictly excluded using a 25% dilated `odMask` to prevent false positive exudate counts on the optic nerve head.
- **Hemorrhage Candidates**: Segments dark intra-retinal blotches ($40\text{--}2500\text{ px}$).
- **Neovascularization Risk**: Calculates a non-diagnostic risk indicator (`Low` / `Moderate` / `High`) derived from peri-papillary vessel density and vascular branching.

*All lesion findings are explicitly labeled as computer-vision candidate detections, never autonomous diagnoses.*

---

## 8. DR Severity

The primary deep learning model classifies the scan into the 5-stage International Clinical Diabetic Retinopathy (ICDR) scale:
- **Grade 0**: No Diabetic Retinopathy
- **Grade 1**: Mild Non-Proliferative DR
- **Grade 2**: Moderate Non-Proliferative DR
- **Grade 3**: Severe Non-Proliferative DR
- **Grade 4**: Proliferative Diabetic Retinopathy

The UI presents the predicted grade, associated clinical severity string, and the softmax prediction confidence percentage.

---

## 9. Referable DR

RetinaGuard applies the international screening threshold:
$$\text{Referable DR} \iff \text{Grade} \ge 2$$

- **Non-Referable (Grades 0 & 1)**: Annual or 6-to-12 month routine monitoring.
- **Referable (Grades 2, 3, & 4)**: Triggers an alert in the UI requiring formal ophthalmology evaluation. On the validation set, EXP-001 achieves **78.86% sensitivity** and **92.64% specificity** for referable disease.

---

## 10. Grad-CAM

Clicking the **Heatmap** button displays the true Gradient-weighted Class Activation Map (Grad-CAM):
- Computes gradients of the target class logit with respect to the final convolutional feature map of EfficientNet-B0.
- Normalizes activations and projects a color-coded thermal map (Jet colormap) directly onto the original fundus photograph.
- Explains *where* the neural network focused its attention (e.g. peri-macular exudates or retinal hemorrhages).
- Labeled in the UI as **"Grad-CAM / Model Attention"** to distinguish it from manual lesion segmentation.

---

## 11. Recommendation

Based on the predicted grade and image quality, the system outputs clear clinical decision-support guidance:
- **Grade 0**: "No signs of DR detected. Routine follow-up in 12 months. Maintain glycemic control."
- **Grade 1**: "Mild non-proliferative DR. Clinical review in 6–12 months."
- **Grade 2**: "Moderate non-proliferative DR. Ophthalmology evaluation recommended within 3–6 months."
- **Grade 3**: "Severe non-proliferative DR. Ophthalmology referral required within 1 month."
- **Grade 4**: "Proliferative DR. Urgent referral to retina specialist required."
- **Ungradable**: "Image quality insufficient for grading. Recapture retinal image with improved focus and illumination."

---

## 12. Report

Clicking **Generate Report** produces a self-contained HTML clinical report in `results/reports/`:
- Formats patient demographics, examination timestamp, and eye orientation.
- Quality score bar and sub-metrics.
- Severity grade badge with confidence level.
- Quantitative lesion candidate table and anatomical landmark measurements.
- Prominent medical disclaimer stating the report is an AI-assisted screening triage aid requiring specialist validation.
- Automatically opens in the system's default web browser.

---

## 13. Demo Mode

The application includes six curated demonstration cases:
1. **DEMO-001 (Normal Retina — No DR)**: Demonstrates clean fundus scan, Grade 0 prediction, routine 12-month follow-up.
2. **DEMO-002 (Mild NPDR)**: Demonstrates early microvascular changes and non-referable triage.
3. **DEMO-003 (Moderate NPDR)**: Demonstrates hard exudates, referable triage gate, and 3–6 month referral.
4. **DEMO-004 (Severe NPDR)**: Demonstrates extensive hemorrhages and 1-month referral recommendation.
5. **DEMO-005 (Proliferative DR)**: Demonstrates advanced disease presentation, high lesion burden, and urgent referral.
6. **DEMO-006 (Poor Quality — Ungradable)**: Demonstrates the safety quality gate stopping downstream grading on severe blur/underexposure.

---

## 14. Error Handling

The application provides graceful error handling:
- **Corrupt / Non-Image Files**: Caught safely with an informative alert dialog; does not crash the UI.
- **Severely Degraded Images**: Intercepted by `assessImageQuality`, marked `UNGRADABLE`, halting downstream classification.
- **Extreme Brightness / Darkness**: Flagged by mean illumination threshold checks.
- **Missing Models**: Automatically falls back to the embedded rule-based heuristic with an explicit warning banner.

---

## 15. Limitations

- **Computer Vision Candidates vs Trained Detectors**: Lesion candidate counts (microaneurysms, exudates, hemorrhages) are derived from mathematical morphology and intensity clustering, not deep learning segmentation models. They are designed to highlight regions of interest for human review.
- **Grade 4 Recall**: As documented in the EXP-001 scorecard, PDR represents ~1.9% of the training dataset, resulting in a 15.25% recall for Grade 4 specifically. Any Grade $\ge 2$ triggers the referable pathway, ensuring patient safety.
- **Regulatory Status**: RetinaGuard is an educational and hackathon research prototype. It has not undergone multi-center clinical trials and must not be used as an autonomous medical diagnostic device.

---

## 16. SIH Demonstration Flow (3–5 Minute Presentation)

1. **Step 1 — Launch & Introduction (30s):**
   - Execute `run('launch.m')` in MATLAB.
   - Introduce RetinaGuard as an edge-ready, offline-first screening workstation for rural clinics.
2. **Step 2 — Run Standard Screening (Case 3 — Moderate NPDR) (60s):**
   - Select *Moderate Non-Proliferative DR* $\rightarrow$ Click **Load Demo**.
   - Click **Analyze Image ▶** (runs in $< 1$ second).
   - Point out Quality Score (82/100, GOOD), detected exudates/MAs, vessel density, and Grade 2 result.
   - Emphasize the **Referable DR** flag and 3–6 month referral timeline.
3. **Step 3 — Demonstrate Explainability & Layer Toggles (45s):**
   - Click **Enhanced** to show CLAHE contrast equalization.
   - Click **Vessels** to show segmented vascular tree.
   - Click **Lesions** to show color-coded candidate markers.
   - Click **Heatmap** to demonstrate real **Grad-CAM** saliency highlighting the pathology.
4. **Step 4 — Demonstrate Patient Safety Quality Gate (Case 6) (45s):**
   - Select *Poor Quality — Ungradable* $\rightarrow$ Click **Load Demo** $\rightarrow$ Click **Analyze Image ▶**.
   - Show that the system flags **UNGRADABLE**, halts grading, and displays actionable recapture instructions.
5. **Step 5 — Generate Report & Wrap Up (30s):**
   - Click **Generate Report**; show the generated HTML report.
   - Highlight offline capability, locked EXP-001 model ($\text{QWK} = 0.7342$), and clinical disclaimer.
