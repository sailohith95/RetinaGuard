# RetinaGuard — System Architecture

## Overview

RetinaGuard is structured as a **layered MATLAB application** that separates presentation, pipeline logic, model inference, and data storage into cleanly defined boundaries. This allows real trained models to replace demo placeholders without modifying any UI code.

---

## Layer Model

```
┌─────────────────────────────────────────────────────────┐
│             PRESENTATION LAYER                          │
│   RetinaGuardApp.m  — MATLAB uifigure-based UI         │
│   6 navigation modules: Screening / Patients /          │
│   Examinations / Reports / Model / System               │
└─────────────────────┬───────────────────────────────────┘
                      │  calls
┌─────────────────────▼───────────────────────────────────┐
│             APPLICATION LOGIC LAYER                     │
│   runScreeningPipeline.m — 5-stage orchestrator        │
│   getDemoCases.m          — demo case definitions      │
│   generateReport.m        — HTML report builder        │
└──┬──────────┬──────────┬──────────┬────────────────────┘
   │          │          │          │  calls
┌──▼──┐  ┌───▼──┐  ┌────▼──┐  ┌───▼────────────────────┐
│     │  │      │  │       │  │                         │
│Quality│ │Enh-  │  │Struct-│  │  Lesion    │  Grading   │
│Assess│  │ance  │  │ures   │  │  Detection │  (gradeDR) │
│      │  │      │  │       │  │            │            │
└──┬──┘  └───┬──┘  └────┬──┘  └────┬───────┴────┬───────┘
   │          │          │          │             │
┌──▼──────────▼──────────▼──────────▼─────────────▼──────┐
│             MODEL INTERFACE LAYER                        │
│   DRClassifierModel.m  — loads .mat or uses heuristic  │
│   (QualityModel, LesionModel, StructureModel stubs)    │
└─────────────────────┬───────────────────────────────────┘
                      │  reads
┌─────────────────────▼───────────────────────────────────┐
│             DATA / MODEL STORAGE LAYER                   │
│   models/dr/dr_classifier.mat  (absent → demo mode)   │
│   data/aptos/train_images/     (absent → no training)  │
│   data/idrid/                  (absent → no lesions)   │
│   config/RGConfig.m            (all paths here)        │
└─────────────────────────────────────────────────────────┘
```

---

## Component Reference

### `app/RetinaGuardApp.m`
The single MATLAB `matlab.apps.AppBase` subclass that owns the entire UI. Implements:
- `buildUI()` — constructs all panels at startup
- `navigateTo(section)` — shows/hides panels
- `onAnalyze()` — calls `runScreeningPipeline` or demo fast-path
- `populateQuality/Findings/Structures/Grading()` — updates display widgets
- `drawSeverityScale(grade)` — renders 5-cell ICDR bar
- `onGenerateReport()` — calls `generateReport` and opens HTML in browser

### `config/RGConfig.m`
Single source of truth for all configurable values:
- File paths (dataset dirs, model dirs, report dir)
- Preprocessing parameters (target size, CLAHE clip limit, noise sigma)
- Quality thresholds (min score, Laplacian variance floor)
- Confidence thresholds (high/moderate/low boundaries)
- DR grade labels and referral threshold

**To change any path or threshold, edit only `RGConfig.m`.**

### `core/screening/runScreeningPipeline.m`
Orchestrates the 5-stage pipeline:
1. `assessImageQuality` — if ungradable, stops here
2. `enhanceRetinalImage` — preprocessing chain
3. `detectStructures` — OD, fovea, vessels
4. `detectLesions` — MA, exudate, hemorrhage, NV
5. `gradeDR` — severity grade + confidence + heatmap

Returns a single `result` struct covering all stages.

### `core/quality/assessImageQuality.m`
Computes quality sub-scores:
- **Focus**: Laplacian variance of the green channel
- **Illumination**: mean green-channel pixel intensity (checks under/overexposure)
- **Field of View**: fraction of non-black pixels relative to frame area
- **Contrast**: standard deviation of green channel

Aggregates into a 0–100 score. Below `cfg.quality.minScore` (default 40) → `UNGRADABLE`.

### `core/enhancement/enhanceRetinalImage.m`
Sequential preprocessing:
1. Grayscale/RGBA normalization → uint8 RGB
2. `removeBlackBorder` → tight crop around retinal disc
3. `imresize` to `cfg.preprocess.targetSize` (default 512×512)
4. Illumination normalization (subtract large-sigma Gaussian background)
5. Per-channel CLAHE (`adapthisteq`)
6. Gaussian denoising (`imgaussfilt`)
7. Per-channel min-max intensity stretch

Original image is **never modified**. Enhanced image returned separately.

### `core/structures/detectStructures.m`
Computer-vision based anatomical landmark localization:
- **Optic Disc**: Peak luminance detection with morphological refinement; outputs centroid `[odRow, odCol]`, radius `odR`, and explicit circular binary mask `odMask`.
- **Fovea Centralis**: Geometric projection from optic disc (~2.5 disc diameters temporal) refined by local luminance minimum search; outputs coordinates and bounding radius.
- **Retinal Vessels**: Multi-scale Gaussian ridge filtering (simplified Frangi vesselness) combined with morphological top-hat; computes vascular binary mask and `vessels.density` (% of total retinal area).

Generates multi-color overlay images with yellow circle for OD, cyan cross for fovea, and green vascular highlight.

### `core/lesions/detectLesions.m`
Quantitative computer-vision lesion candidate analysis:
- **Microaneurysms**: Morphological top-hat filtering on inverted green channel targeting small isolated circular dark blobs (3–65 px); returns candidate count and mask.
- **Hard Exudates**: High-luminance intra-retinal lipid deposition segmentation. **Crucially, the optic disc region is masked out using dilated `odMask`** to avoid false positives on the optic nerve head; returns candidate count and area.
- **Hemorrhages**: Contrast-enhanced dark intra-retinal lesion segmentation (40–2500 px); returns candidate count and area.
- **Neovascularization Risk Indicator**: Non-diagnostic risk metric (`Low` / `Moderate` / `High`) derived from peri-papillary vessel density and vascular branching.

Returns per-lesion candidate binary masks, quantitative metrics, and a composite multi-color diagnostic overlay.

### `core/grading/gradeDR.m` & `core/grading/runPythonInference.m`
Multi-backend production grading orchestrator:
1. **ONNX Runtime (`runONNXModel.m`)**: Loads `models/aptos_efficientnet/best_model.onnx` directly into MATLAB for high-speed local inference.
2. **Python Bridge (`runPythonInference.m`)**: Invokes Python CLI (`predict.py --gradcam`) to run locked **EXP-001** PyTorch weights and retrieve true Grad-CAM saliency heatmaps.
3. **Fallback Heuristic**: Used only if neither model format is available.

Outputs 5-class probability vector, ICDR grade, referable status, clinical recommendation, and Grad-CAM heatmap.

### `demo/getDemoCases.m` & `demo/sample_images/`
Provides 6 curated screening cases:
- Cases 1–5: Authentic clinical fundus scans representing Grades 0 through 4 (No DR, Mild, Moderate, Severe, PDR).
- Case 6: Authentic degraded fundus image demonstrating image quality rejection (Ungradable).
- Algorithmic synthetic generator `generateSyntheticFundus.m` serves as offline fallback.

---

## Model Interface Contracts

All model functions share a consistent interface so the UI never needs to change when real models are added.

### `predictDR(img)` → `result`
```
result.grade          — integer 0–4
result.confidence     — scalar 0–1
result.probabilities  — 1×5 class probabilities
result.label          — full severity string
result.referralText   — clinical recommendation string
result.heatmap        — uint8 RGB saliency overlay
result.demoMode       — logical
```

### `assessImageQuality(img, cfg)` → `result`
```
result.score         — 0–100
result.status        — 'GOOD' | 'BORDERLINE' | 'UNGRADABLE'
result.focus         — 'Good' | 'Acceptable' | 'Poor'
result.illumination  — 'Good' | 'Acceptable' | 'Poor'
result.fieldOfView   — 'Good' | 'Acceptable' | 'Poor'
result.gradable      — logical
result.reason        — string (non-empty when ungradable)
result.recommendation — string
```

### `detectLesions(img, cfg)` → `result`
```
result.microaneurysm.detected / .confidence / .mask / .count
result.exudate.detected       / .confidence / .mask / .count
result.hemorrhage.detected    / .confidence / .mask / .count
result.neovascularization.detected / .confidence / .mask
result.heatmap   — uint8 RGB lesion overlay
result.demoMode  — logical
```

### `detectStructures(img, cfg)` → `result`
```
result.opticDisc.detected / .confidence / .centroid / .overlay
result.fovea.detected     / .confidence / .centroid
result.vessels.detected   / .confidence / .mask / .overlay
result.demoMode — logical
```

---

## Design Decisions

### Offline-First
The application runs entirely within MATLAB without any internet connection or external API. This supports deployment in rural screening camps and low-connectivity environments.

### Demo Mode
When `models/dr/dr_classifier.mat` is absent, every result is labelled `demoMode = true`. The UI renders a persistent amber banner: **"DEMO MODE — Results are not medically validated."** This distinction is enforced at every level (pipeline, report, UI labels).

### Model Replacement Strategy
Replacing any heuristic module with a trained model requires only:
1. Save the trained network: `save(cfg.paths.drModel, 'net', 'classes')`
2. Restart the app (or call `tryLoad` on the model class)

Zero UI changes are needed. The pipeline detects the model file automatically.

### No Fake Clinical Claims
- No invented sensitivity/specificity/AUC numbers
- No "FDA approved" or "clinically validated" language
- Demo predictions explicitly labelled as non-clinical
- Report includes mandatory disclaimer on every generated document

---

## Future Extension Points

| Capability | Where to Add |
|---|---|
| Real DR classifier | Replace heuristic in `gradeDR.m`, save `.mat` to `models/dr/` |
| Lesion segmentation (IDRiD) | Replace heuristic in `detectLesions.m` |
| Structure detection (DL) | Replace heuristic in `detectStructures.m` |
| PDF report export | Add MATLAB `print` call in `generateReport.m` |
| Patient database persistence | Add SQLite or CSV store to `app/RetinaGuardApp.m` Patients module |
| Batch screening mode | New script in `core/screening/` using `batchPreprocess.m` |
| DICOM support | Add `dicomread` branch in image loading callback |
| Telemedicine upload | Add REST call after report generation |
