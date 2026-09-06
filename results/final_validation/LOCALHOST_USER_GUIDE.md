# RetinaGuard — Localhost Web Application User Guide

> **Step-by-Step Operator & Demonstration Guide**  
> *How to launch, operate, and present the RetinaGuard AI-assisted diabetic retinopathy screening workstation in a web browser.*

---

## 1. Prerequisites

- Python 3.9, 3.10, or 3.11 installed.
- Standard terminal (PowerShell, Command Prompt, or Bash).
- Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Safari).
- **Zero Internet Connection Required**: The system operates 100% locally and offline.

---

## 2. Launching the Workstation

### Step 1: Open Terminal in Project Directory
Open your terminal and navigate to the project root:
```bash
cd "c:\Users\T SAILOHITH\Desktop\SIH\RetinaGuard"
```

### Step 2: Execute the Single Launch Command
Run the unified launcher:
```bash
python webapp/run.py
```

### Step 3: Verify Terminal Startup Output
You will see the RetinaGuard ASCII banner and service initialization:
```
======================================================================
           RETINAGUARD - AI SCREENING WORKSTATION
           Localhost Web Application Launcher
======================================================================
[INFO] Model Checkpoint: models/aptos_efficientnet/best_model.pt
[INFO] Architecture: EfficientNet-B0 (EXP-001 Verified)
[INFO] Operating Mode: 100% Offline & Air-Gapped Localhost
[INFO] Launching server on http://localhost:8000 ...
[INFO] Default web browser opened automatically.
======================================================================
```

### Step 4: Browser Navigation
Your default browser will automatically open to `http://localhost:8000`.  
*(If your browser does not open automatically, simply copy and paste `http://localhost:8000` into your address bar).*

---

## 3. Workstation Interface Tour

The workstation is organized into a dual-column medical imaging layout:

1. **Top Navigation Bar**: Displays system status (`Production Model: EXP-001 (EfficientNet-B0)`), offline status pill (`Offline / Localhost`), and quick links.
2. **Left Panel — Retina Viewport**:
   - High-resolution interactive canvas.
   - **Layer Selector Tabs**: Seamlessly toggle between:
     - `Original`: Unaltered fundus image.
     - `Enhanced`: Green-channel CLAHE contrast enhancement with circular FOV crop.
     - `Vessels`: Extracted vascular tree with computed vascular density percentage.
     - `Lesions`: Composite overlay with color-coded candidate lesions (red = microaneurysms, yellow = exudates, cyan = hemorrhages).
     - `Heatmap`: Real gradient-weighted class activation mapping (Grad-CAM).
   - Dynamic visual legend corresponding to the active layer.
3. **Right Panel — Diagnostic Decision Support**:
   - **Case Ingestion**: File upload dropzone and **Demo Cases Dropdown**.
   - **Quality Assessment Card**: Composite quality score ($0\text{--}100$), sharpness, illumination, contrast, and FOV %.
   - **Safety Gate Alert**: Renders an alert box when poor quality halts grading.
   - **DR Severity Grading Card**: Predicted ICDR Grade ($0\text{--}4$), diagnostic category, softmax confidence %, and probability distribution bars.
   - **Referral Triage Badge**: High-visibility status indicator (`NON-REFERABLE` vs `REFERABLE DR`).
   - **Candidate Lesion Summary**: Quantitative counts of candidate lesions and neovascularization risk.
   - **Export Report Button**: One-click download of the complete clinical screening report.
   - **Investigational Disclaimer**: Explicit medical device decision-support disclaimer.

---

## 4. Step-by-Step Demonstration Script

### Demo 1: Normal Retina Screening (Grade 0)
1. In the **"Select Demo Case"** dropdown, pick **"Case 1 — Normal Retina (No DR)"**.
2. Click **"Analyze Retinal Image"** (or observe automatic loading).
3. **Inspect Results**:
   - **Quality Score**: High quality score (> 70/100, `GOOD`).
   - **DR Grade**: `Grade 0 — No Diabetic Retinopathy`.
   - **Referral Status**: Green badge: **NON-REFERABLE**.
   - **Clinical Recommendation**: *"No diabetic retinopathy detected. Routine annual retinal screening recommended in 12 months."*
4. **Inspect Layers**:
   - Click **"Vessels"**: Notice clean vascular arborization without neovascular fronds.
   - Click **"Lesions"**: 0 candidate exudates or hemorrhages detected.

### Demo 2: Mild NPDR Reference (Grade 1)
1. Select **"Case 2 — Mild NPDR Reference"** from the dropdown.
2. **Inspect Results**:
   - **DR Grade**: `Grade 1 — Mild Non-Proliferative DR`.
   - **Referral Status**: Green badge: **NON-REFERABLE (EARLY STAGE)**.
   - **Clinical Recommendation**: Re-screen in 6–12 months; strict glycemic and blood pressure management.
3. **Inspect Layers**:
   - Click **"Lesions"**: Observe isolated candidate microaneurysms highlighted with red markers.

### Demo 3: Moderate NPDR with Referral Triage (Grade 2)
1. Select **"Case 3 — Moderate NPDR Reference"**.
2. **Inspect Results**:
   - **DR Grade**: `Grade 2` or `Grade 3`.
   - **Referral Status**: Orange/Red badge: **REFERABLE DR**.
   - **Clinical Recommendation**: *"Prompt referral to an ophthalmologist within 2-4 weeks for comprehensive dilated fundus examination and OCT evaluation."*
3. **Inspect Explainability**:
   - Click **"Heatmap (Grad-CAM)"**: Point out to observers that the neural network's high attention regions correspond directly to the anatomical clusters of hemorrhages and exudates.

### Demo 4: Proliferative DR Reference (Grade 4)
1. Select **"Case 5 — Proliferative DR Reference"**.
2. **Inspect Results**:
   - **DR Grade**: `Grade 4 — Proliferative Diabetic Retinopathy`.
   - **Referral Status**: Urgent Red badge: **URGENT REFERRAL REQUIRED**.
   - **Clinical Recommendation**: *"Urgent ophthalmology referral within 24–48 hours for immediate evaluation (panretinal photocoagulation / anti-VEGF therapy consideration)."*
3. **Inspect Lesion Findings**:
   - Observe candidate hemorrhage counts and elevated neovascularization risk indicator.

### Demo 5: Testing the Ungradable Safety Gate (Critical Medical AI Feature)
1. Select **"Case 6 — Poor Quality / Ungradable"**.
2. Click **"Analyze Retinal Image"**.
3. **Observe the Safety Gate in Action**:
   - The **Quality Score** drops to **10.7 / 100** (`UNGRADABLE`).
   - The **Safety Gate Alert** immediately appears in bold warning styling:
     > **⚠️ CLINICAL SAFETY GATE TRIGGERED: IMAGE UNGRADABLE**  
     > *Diagnostic grading has been halted to prevent false negative clinical decisions.*  
     > **Recapture Action**: *Recapture retinal image with improved focus, centered illumination, and full pupil dilation.*
   - Note that **DR Grading is NOT performed** on this corrupted image, preventing catastrophic diagnostic errors.

### Demo 6: Clinical Report Generation & Export
1. Return to any gradable case (e.g., Demo Case 3).
2. Click the blue **"📄 Download Clinical Report"** button in the bottom-right panel.
3. A complete, standalone HTML report file will download instantly (e.g., `report_EX-00412_20260906.html`).
4. Open the downloaded file in your browser to showcase the print-ready medical layout:
   - Patient metadata header.
   - 5-panel side-by-side diagnostic thumbnails.
   - Comprehensive quantitative tables (sharpness, illumination, OD coordinates, vessel density, lesion counts).
   - Triage outcome and follow-up directives.
   - Formal clinical decision support audit trail and disclaimer.

### Demo 7: Custom Image Upload
1. Drag and drop any `.png` or `.jpg` fundus image onto the upload box, or click the file picker.
2. The workstation instantly ingests, processes, and displays the full diagnostic breakdown.

---

## 5. Clean Shutdown

When finished with the demonstration:
1. Switch back to your terminal window.
2. Press `Ctrl + C`.
3. The Uvicorn server will terminate cleanly and release the port.

---

## 6. Troubleshooting

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| **Port 8000 already in use** | Another local process is using 8000 | `run.py` automatically detects this and binds to port 8001, 8002, etc. Check terminal output for exact URL. |
| **Browser doesn't open automatically** | OS policy or headless session | Manually navigate to `http://localhost:8000` in Chrome, Edge, or Firefox. |
| **"Ungradable" alert appears on good image** | Poor illumination or off-center FOV | Ensure the fundus image contains a visible circular retina with adequate focus. |
| **Need to test without browser** | Headless automated verification | Run `python webapp/test_live_server.py` to verify full functionality via CLI. |
