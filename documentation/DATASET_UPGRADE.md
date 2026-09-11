# RetinaGuard — Autonomous Fundus Dataset Upgrade & Audit Report

**Dataset Name**: Indian Diabetic Retinopathy Image Dataset (IDRiD)  
**Challenge**: ISBI 2018 Diabetic Retinopathy: Segmentation and Grading Challenge  
**Official Source**: IEEE DataPort & Grand Challenge (ISBI 2018)  
**Research Mirror**: Hugging Face `MahsaTorki/IDRiD_Dataset` (`A.Segmentation.zip`)  
**Audit Date**: 2026-09-11  
**Auditor**: RetinaGuard Automated Pipeline Audit Engine

---

## 1. Executive Summary & Verification

- **APTOS 2019 Status**: Explicitly confirmed as color fundus photographs acquired across rural Indian eye clinics. **EXP-001 (EfficientNet-B0)** remains locked and untouched as the production DR severity classifier.
- **Acquisition**: IDRiD Part A (Segmentation Challenge) was downloaded directly via chunked HTTP stream ($557.25\text{ MB}$) and unpacked into `data/idrid/A. Segmentation/`.
- **Integrity**: Exactly 81 color fundus images were audited. **0 corrupt files**, **0 truncated images**, **0 dimension mismatches**.
- **Split Preservation**: The official ISBI 2018 benchmark split is strictly preserved:
  - **Training Set**: 54 images (`IDRiD_01` to `IDRiD_54`)
  - **Testing Set**: 27 images (`IDRiD_55` to `IDRiD_81`)
  - **Leakage**: **Zero test set leakage**. The test set remains untouched during training and tuning.

---

## 2. Dataset Audit Metrics

### Image Specifications
- **Sensor**: Kowa VX-10alpha digital fundus camera
- **Field of View (FOV)**: 50 degrees
- **Acquisition Site**: Eye clinic in Nanded, Maharashtra, India (rural/semi-urban clinical population)
- **Image Resolution**: $4288 \times 2848$ pixels (uniform across all 81 images)
- **Color Space**: 24-bit RGB (3 channels)

---

## 3. Ground Truth Annotations Breakdown

Every annotation consists of an expert-drawn pixel-level binary mask:
- Stored as Indexed/Palette TIFF (`uint8`), where `0 = Background` and `1 = Positive Pixel`.
- In IDRiD, missing mask files represent negative cases (i.e. zero lesion burden of that class in that fundus scan).

### Training Set ($N = 54$ Images)

| Structure / Lesion | Category | Available Masks | Positive Images | Empty / Negative Images | Positive Rate (%) | Mean Pixel Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Microaneurysms (MA)** | Pathology | 54 | 54 | 0 | 100.0% | 0.1069% |
| **Haemorrhages (HE)** | Pathology | 53 | 53 | 1 | 98.15% | 1.0043% |
| **Hard Exudates (EX)** | Pathology | 54 | 54 | 0 | 100.0% | 0.8066% |
| **Soft Exudates (SE)** | Pathology | 26 | 26 | 28 | 48.15% | 0.3934% |
| **Optic Disc (OD)** | Normal Anatomy | 54 | 54 | 0 | 100.0% | 1.8055% |

### Testing Set ($N = 27$ Images, Official Unseen Evaluation Split)

| Structure / Lesion | Category | Available Masks | Positive Images | Empty / Negative Images | Positive Rate (%) | Mean Pixel Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Microaneurysms (MA)** | Pathology | 27 | 27 | 0 | 100.0% | 0.0982% |
| **Haemorrhages (HE)** | Pathology | 27 | 27 | 0 | 100.0% | 1.0665% |
| **Hard Exudates (EX)** | Pathology | 27 | 27 | 0 | 100.0% | 1.0851% |
| **Soft Exudates (SE)** | Pathology | 14 | 14 | 13 | 51.85% | 0.3485% |
| **Optic Disc (OD)** | Normal Anatomy | 27 | 27 | 0 | 100.0% | 1.7313% |

---

## 4. Key Architectural Insights & Decisions

1. **Multi-Label vs. Mutually Exclusive**:
   - Analysis revealed non-zero overlapping pixels between lesions (e.g. 19 pixels between MA & HE, 503 pixels between HE & EX).
   - Therefore, a mutually exclusive Softmax is medically and mathematically invalid. **Sigmoid activation per target channel is mandatory**.
2. **Decoupling Anatomy from Pathology**:
   - **Optic Disc is normal ocular anatomy**, NOT a diabetic retinopathy lesion.
   - Furthermore, Optic Disc occupies 18× more surface area than Microaneurysms (1.81% vs 0.11%).
   - To prevent the large Optic Disc gradient from drowning out tiny microvascular lesion signals, the network employs a **Dual-Head U-Net**:
     - **Head 1 (`lesion_head`)**: 4 channels for Diabetic Retinopathy lesions (`MA`, `HE`, `EX`, `SE`).
     - **Head 2 (`anatomy_head`)**: 1 channel for Optic Disc anatomical localization (`OD`).
3. **Clinical Research Prototype Notice**:
   - The training set contains 54 expert-annotated fundus photographs. The model is an engineering/research prototype designed for decision support and triage assistance, not autonomous medical diagnosis.

---

## 5. Audit Conclusion

All 81 images and 463 total files in the IDRiD Part A dataset are verified and consistent. No corrupt files or dimension anomalies exist. The dataset is approved for model training.
