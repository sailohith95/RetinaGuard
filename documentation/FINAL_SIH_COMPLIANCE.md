# RetinaGuard — Final SIH 2026 Problem Statement Compliance Report

## Executive Overview
RetinaGuard is an AI-assisted diabetic retinopathy (DR) screening system built to satisfy the SIH 2026 problem statement. This document provides the engineering audit trail, verifying each requirement with its code location, methodology, and empirical test results.

---

## Technical Audit & Verification Summary

### 1. Image Quality Assessment & Ungradable Handling
- **Implementation:** `core/quality/assessImageQuality.m`
- **Methodology:** Multi-factor quality metric computing Laplacian gradient variance (sharpness), mean intensity (exposure), Michelson contrast, and SNR.
- **Decision Boundary:** Quality score $\ge 60 \rightarrow$ Good, $40\text{--}59 \rightarrow$ Borderline, $<40 \rightarrow$ Ungradable.
- **Ungradable Action:** Halts downstream grading pipeline, marks case as Ungradable, and outputs actionable recapture guidance (focus and illumination advice).
- **Verification:** Unit test `test_01_image_quality_assessment` and `test_02_ungradable_detection` in `python/test_end_to_end_sih.py`.

### 2. Retinal Image Preprocessing & Contrast Enhancement
- **Implementation:** `core/enhancement/enhanceRetinalImage.m`, `python/preprocessing/retinal_preprocessor.py`
- **Methodology:** Black border removal via binary luminance thresholding, green-channel extraction (highest absorption contrast for hemoglobin/melanin), Contrast Limited Adaptive Histogram Equalization (CLAHE, clip limit = 2.0, grid = $8\times 8$), and median noise filtering.
- **Verification:** `test_03_clahe_enhancement`.

### 3. Anatomical Landmark Localization
- **Optic Disc:**
  - `core/structures/detectStructures.m`
  - Methodology: Luminance morphological filtering to find the highest-intensity circular cluster, computing optic disc centroid $(c_y, c_x)$, radius $r_{od}$, and generating an explicit binary circular mask `odMask`.
- **Fovea Centralis:**
  - `core/structures/detectStructures.m`
  - Methodology: Anatomical geometric projection based on retinal geometry (~2.5 disc diameters temporal to optic disc centroid) refined by local luminance minimum search.
- **Retinal Vasculature & Density:**
  - `core/structures/detectStructures.m`
  - Methodology: Multi-scale Gaussian ridge filtering + morphological black top-hat filter. Computes retinal vascular density:
    $$\text{Vessel Density (\%)} = \frac{\sum (\text{Vessel Mask} \land \text{Retina Mask})}{\sum \text{Retina Mask}} \times 100$$
- **Verification:** `test_04_optic_disc_detection`, `test_05_fovea_localization`, `test_06_vessel_segmentation_and_density`.

### 4. Computer Vision Lesion Candidate Analysis
- **Hard Exudates:**
  - `core/lesions/detectLesions.m`
  - Methodology: High-luminance intra-retinal lipid deposition segmentation. Crucially, the optic disc region is masked using a $25\%$ dilated `odMask`, eliminating false-positive exudate detections on the optic nerve head. Reports candidate count and area.
- **Microaneurysms:**
  - `core/lesions/detectLesions.m`
  - Methodology: Green-channel morphological top-hat candidate detection targeting $3\text{--}65\text{ px}$ circular dark focal lesions.
- **Hemorrhages:**
  - `core/lesions/detectLesions.m`
  - Methodology: Contrast-enhanced dark intra-retinal lesion segmentation ($40\text{--}2500\text{ px}$).
- **Neovascularization Risk:**
  - `core/lesions/detectLesions.m`
  - Methodology: Non-diagnostic risk metric (Low/Moderate/High) calculated from peri-papillary vessel density and vascular branching.
- **Verification:** `test_07_hard_exudates_candidate_detection`, `test_08_microaneurysm_candidates`, `test_09_hemorrhage_candidates`, `test_10_neovascularization_risk_indicator`.

### 5. Primary Production Model (EXP-001 Lock)
- **Model Checkpoints:** `models/aptos_efficientnet/best_model.pt` & `models/aptos_efficientnet/best_model.onnx`
- **Architecture:** EfficientNet-B0 backbone with Dropout ($p=0.4$) and a 5-unit linear classification head.
- **Performance:**
  - Quadratic Weighted Kappa (QWK): **0.7342**
  - Accuracy: **69.85%**
  - Macro F1: **0.4981**
  - Referable Sensitivity (Grade $\ge 2$): **78.86%**
  - Referable Specificity (Grade $< 2$): **92.64%**
- **ONNX Runtime Parity:** Max absolute difference between PyTorch and ONNX logits is $4.05 \times 10^{-6}$ ($< 1 \times 10^{-4}$).
- **Verification:** `test_11_pytorch_exp001_inference`, `test_12_onnx_exp001_parity`.

### 6. Explainability (True Grad-CAM)
- **Implementation:** `python/explainability/gradcam.py`, `core/grading/runPythonInference.m`
- **Methodology:** Gradient-weighted Class Activation Mapping computed from the final convolutional stage of EfficientNet-B0. Generates target-class spatial activations and blends them with the original fundus photograph using an alpha colormap overlay.
- **Verification:** `test_13_gradcam_generation`.

### 7. Clinical Decision Support & Reporting
- **Implementation:** `core/reporting/generateReport.m`, `python/inference/predict.py`
- **Output:** Professional HTML screening report containing patient/exam metadata, image quality scores, DR grade, model confidence, lesion candidate statistics, anatomical landmark measurements, referral priority, follow-up timeline, and medical disclaimer.
- **Verification:** `test_14_clinical_decision_support`, `test_15_html_screening_report_generation`.

### 8. Workstation Application & Curated Demo Mode
- **Implementation:** `app/RetinaGuardApp.m`, `demo/getDemoCases.m`, `demo/sample_images/`
- **Capabilities:** Interactive UI with original/enhanced image views, multi-layer overlays (optic disc, fovea, vessels, lesions, heatmap), and 6 authentic demo cases (Grades 0–4 + Ungradable) extracted from authentic clinical fundus scans.
- **Verification:** `test_16_demo_cases_integrity`.
