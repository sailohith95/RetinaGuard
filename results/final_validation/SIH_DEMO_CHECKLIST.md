# RetinaGuard — SIH 2026 Live Demonstration Checklist

Use this checklist during live demonstration to evaluators and technical judges:

- [ ] **MATLAB launched**: Executed `run('launch.m')` in MATLAB command window without errors.
- [ ] **Demo case loaded**: Selected case from dropdown (e.g. *Moderate Non-Proliferative DR*) and clicked **Load Demo**.
- [ ] **Image displayed**: Central image viewport renders the fundus photograph clearly.
- [ ] **Quality assessment shown**: Quality score (0–100) and status badge (`GOOD` / `BORDERLINE` / `UNGRADABLE`) populated.
- [ ] **Enhancement shown**: Toggled **Enhanced** button; verified CLAHE contrast enhancement and black border cropping.
- [ ] **Optic disc shown**: Optic disc localized with yellow circle overlay, centroid, radius, and mask.
- [ ] **Fovea shown**: Fovea centralis localized with cyan cross overlay and coordinates displayed.
- [ ] **Vessel overlay shown**: Toggled **Vessels** button; verified green vascular segmentation and vessel density % display.
- [ ] **Lesion overlay shown**: Toggled **Lesions** button; verified color-coded candidate markers (yellow=exudates, magenta=MAs, red=hemorrhages).
- [ ] **DR Grade shown**: Deep learning model prediction displayed across 5-class ICDR scale (Grades 0–4).
- [ ] **Confidence shown**: Softmax model probability percentage displayed alongside confidence badge.
- [ ] **Referable status shown**: Referable DR flag triggered appropriately for Grade $\ge 2$.
- [ ] **Grad-CAM shown**: Toggled **Heatmap** button; verified true backward-pass saliency map overlay on fundus image.
- [ ] **Recommendation shown**: Actionable clinical recommendation displayed (routine follow-up vs specialist referral).
- [ ] **Report generated**: Clicked **Generate Report** button; verified HTML document creation in `results/reports/`.
- [ ] **Report opened**: HTML report renders in web browser with all quantitative tables and metrics.
- [ ] **Disclaimer visible**: Confirmed presence of medical disclaimer stating system is an AI-assisted screening decision-support aid.
- [ ] **Safety gate demonstrated**: Loaded *Poor Quality — Ungradable* case; confirmed system halts grading and provides recapture instructions.
