# RetinaGuard — Model Evaluation Methodology & Metrics

## Clinical & Technical Metrics

Diabetic Retinopathy (DR) grading on the 5-class International Clinical Diabetic Retinopathy (ICDR) scale is an **ordinal classification** problem. Accuracy alone is an insufficient metric due to class imbalance and clinical severity progression.

The evaluation suite calculates:

### 1. Quadratic Weighted Kappa (QWK)
Cohen's Kappa with quadratic penalty weights:
$$w_{ij} = \frac{(i - j)^2}{(N - 1)^2}$$
Misclassifying Grade 0 as Grade 1 incurs a penalty of $(0-1)^2 = 1$.
Misclassifying Grade 0 as Grade 4 incurs a penalty of $(0-4)^2 = 16$.
QWK is the standard benchmark metric for Diabetic Retinopathy challenges (APTOS, EyePACS).

### 2. Multi-Class Metrics
- **Accuracy**: Overall fraction of correct predictions across all 5 classes.
- **Macro F1**: Unweighted mean of F1-scores across all 5 classes (penalizes poor performance on minority classes).
- **Per-Class Precision, Recall, F1**: Explicitly tracked for:
  - Class 0: No DR
  - Class 1: Mild DR
  - Class 2: Moderate DR
  - Class 3: Severe DR
  - Class 4: Proliferative DR

### 3. Referable DR Screening Metrics
In clinical screening protocols, patients with Grade ≥ 2 (Moderate NPDR, Severe NPDR, or PDR) require referral to an ophthalmologist:
- **Referable Sensitivity**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ for cases with Grade ≥ 2.
- **Referable Specificity**: $\frac{\text{TN}}{\text{TN} + \text{FP}}$ for cases with Grade < 2.

---

## Evaluation Procedure

To evaluate the current model on the held-out validation set:

```bash
python python/training/evaluate.py
```

This generates:
- `results/metrics_summary.json` (Structured metric values)
- `results/confusion_matrix.png` (Visual matrix with normalized row heatmaps)
- `results/classification_report.txt` (Per-class table)
- `models/aptos_efficientnet/metrics.json` (Model release snapshot)

---

## Integrity & Non-Fabrication Rule

RetinaGuard adheres to strict evaluation integrity:
- No metrics are ever hardcoded into UI or reporting modules.
- If a model has not been evaluated, the system displays "Model Not Trained" or enters Demo Mode.
- All evaluation figures reflect actual test runs against verified validation images.
