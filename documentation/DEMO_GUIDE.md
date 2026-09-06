# RetinaGuard — SIH 2026 Demo Guide

## Target Audience
SIH 2026 evaluators, ophthalmology faculty, and technical judges.

## Total Demo Time
**2–3 minutes** (concise and impactful)

---

## Pre-Demo Checklist

- [ ] MATLAB is open
- [ ] Working directory is `RetinaGuard/`
- [ ] `launch.m` has been run at least once to verify no startup errors
- [ ] Screen resolution set to 1920×1080 or higher
- [ ] Font rendering is clear at presentation distance

---

## Demo Script

### Step 1 — Launch (15 seconds)

```matlab
run('launch.m')
```

**Say:** *"RetinaGuard launches directly into the Screening Workstation — the clinician's primary task. There is no marketing landing page."*

Point out:
- Navigation sidebar (6 modules)
- Demo Mode amber banner
- Examination header (patient ID, eye, status)
- Image workspace is the visual centrepiece

---

### Step 2 — Load Demo Case: Moderate NPDR (20 seconds)

Select **"Moderate Non-Proliferative DR"** from the Demo dropdown → click **Load Demo**.

**Say:** *"An authentic clinical retinal fundus image loads instantly from the curated demo suite. In production, the clinician captures or uploads a raw fundus camera scan."*

Point out:
- Retinal disc, blood vessels, optic disc, and macula clearly visible
- Image occupies the majority of the central workspace

---

### Step 3 — Run Analysis (30 seconds)

Click **Analyze Image ▶**.

**Say:** *"The 5-stage pipeline runs automatically — image quality assessment, contrast enhancement, anatomical landmark detection, lesion analysis, and deep learning DR grading via the locked EXP-001 model."*

Watch the progress stages complete in < 1 second.

After completion, walk through each panel:

**Image Quality:**
> *"Score: 82/100. Focus, illumination, and field of view — all Good. The image is gradable."*

**Lesion Candidate Analysis:**
> *"Microaneurysms detected via morphological top-hat filtering. Hard exudates segmented outside the optic disc. Hemorrhages detected. Neovascularization risk indicator: Low."*

**Retinal Structures:**
> *"Optic disc localized with explicit circular mask, fovea coordinates projected, retinal vessel density calculated at ~9.8%."*

**DR Severity Result:**
> *"AI-assisted screening result: Grade 2 — Moderate Non-Proliferative DR. Model confidence: 78.9% (EXP-001). Referable DR flag triggered. Clinical recommendation: Ophthalmology evaluation within 3–6 months."*

---

### Step 4 — Show Overlays (20 seconds)

Click through the overlay buttons:
- **Original** → raw clinical fundus scan
- **Enhanced** → CLAHE-enhanced image (visibly clearer vessels and contrast)
- **Vessel** → green-highlighted segmented retinal vasculature
- **Lesions** → multi-color lesion candidate map (yellow = exudates, magenta = MA, red = hemorrhages)
- **Heatmap** → true Grad-CAM activation heatmap highlighting the anatomical regions driving the neural network decision

**Say:** *"Clinicians can toggle between original, enhanced, segmented vessels, candidate lesions, and real Grad-CAM explainability heatmaps — verifying why the model made its decision."*

---

### Step 5 — Generate Report (20 seconds)

Click **Generate Report**.

**Say:** *"A standardized HTML clinical report is generated instantly — patient ID, examination date, quality score, quantitative lesion metrics, anatomical measurements, DR grade, and referral recommendation."*

HTML report opens in browser. Point out:
- Examination metadata
- Quality section with score bar
- Severity result with colour-coded ICDR grade
- Quantitative lesion candidate table and anatomical landmark measurements
- Clinical disclaimer requiring specialist confirmation

---

### Step 6 — Ungradable Workflow (20 seconds)

Select **"Poor Quality — Ungradable"** from Demo dropdown → Load Demo → Analyze.

**Say:** *"This is a critical patient safety feature. If the image quality is insufficient — here due to severe blur and underexposure — the grading pipeline halts. The system refuses to produce an unreliable grade."*

Point out:
- UNGRADABLE status in quality panel
- Halting of grading model
- Explicit clinical recapture guidance shown automatically to the operator

---

### Step 7 — Model / AI Panel (15 seconds)

Navigate to **Model / AI** in the sidebar.

**Say:** *"The Model panel shows the status of the primary production model — EXP-001 (EfficientNet-B0) trained on the APTOS 2019 dataset. Validated on 733 held-out test scans: QWK = 0.7342, Accuracy = 69.85%, Referable Sensitivity = 78.86%, Specificity = 92.64%, with ONNX runtime parity verified within 4.05 × 10⁻⁶."*

---

### Step 8 — Close with Differentiators (20 seconds)

**Say:**

*"RetinaGuard's key innovations for rural and clinical deployment:"*

1. **Image quality gate** — unreliable images never reach the classifier
2. **Offline-first** — zero internet required; runs locally on standard CPU in < 1.4s
3. **True Grad-CAM Explainability** — full backward-pass saliency visualization
4. **Computer vision lesion candidates** — microaneurysms, disc-excluded hard exudates, hemorrhages
5. **Locked production model (EXP-001)** — QWK = 0.7342 on held-out APTOS validation
6. **Dual MATLAB / Python deployment** — native MATLAB app + ONNX runtime + Python bridge

---

## Common Questions and Answers

**Q: Is this validated clinically?**
A: RetinaGuard is an AI-assisted screening decision-support prototype. It has been empirically validated on the 733-sample held-out APTOS validation set (QWK = 0.7342, Referable Sensitivity = 78.86%, Specificity = 92.64%), but has not undergone multi-center clinical trials. All outputs require ophthalmologist confirmation.

**Q: What dataset was used?**
A: Trained and evaluated on the APTOS 2019 Blindness Detection dataset (3,662 fundus photographs: 2,929 train / 733 validation split with stratified sampling). The demo suite uses authentic clinical fundus scans extracted directly from this benchmark.

**Q: Why MATLAB and Python?**
A: Python provides state-of-the-art PyTorch training and Grad-CAM explainability, while MATLAB provides the clinical workstation GUI (`uifigure`) specified by SIH. They interoperate seamlessly via ONNX and a CLI JSON bridge.

**Q: Can it work on a laptop without a GPU?**
A: Yes. Inference, Grad-CAM generation, and all computer-vision filters run on standard CPU in under 1.4 seconds per examination. GPU is only needed for model training.

**Q: How is patient data protected?**
A: Complete offline-first execution. Images and examination reports are stored strictly on the local workstation; no data is ever transmitted to external cloud servers.
