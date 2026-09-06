# RetinaGuard — Evaluation Methodology

## Overview

This document describes the evaluation framework for RetinaGuard's DR severity classifier and lesion detection models. Evaluation is only meaningful on **held-out test data** that was never seen during training or validation.

> **Current status: No evaluation has been performed yet.**
> These are the planned procedures to execute once the datasets and trained models are available.

---

## Primary Evaluation Script

```matlab
run('evaluation/evaluate_classifier.m')
```

The script computes and displays all metrics described below. It requires:
- `models/dr/dr_classifier.mat` — trained model
- `data/aptos/test.csv` + `data/aptos/test_images/` — held-out test set

---

## Metrics

### 1. Overall Accuracy

```
Accuracy = (correct predictions) / (total predictions)
```

Accuracy alone is **insufficient** for DR screening due to class imbalance. Always report alongside sensitivity and specificity.

### 2. Per-Class Sensitivity (Recall)

```
Sensitivity_k = TP_k / (TP_k + FN_k)
```

The fraction of grade-`k` cases correctly identified. **Critical for screening** — a false negative means a patient with DR is missed.

### 3. Per-Class Specificity

```
Specificity_k = TN_k / (TN_k + FP_k)
```

The fraction of non-grade-`k` cases correctly identified. A false positive leads to unnecessary referral.

### 4. F1 Score (per class)

```
F1_k = 2 × (Precision_k × Sensitivity_k) / (Precision_k + Sensitivity_k)
```

Harmonic mean of precision and recall. Useful for imbalanced classes.

### 5. Confusion Matrix

A 5×5 matrix (rows = actual grade, columns = predicted grade) shows the full error pattern. Key things to check:
- Off-diagonal entries represent misclassifications
- Grade adjacency errors (grade 2 predicted as 3) are less clinically harmful than grade 0 predicted as 4

### 6. Referral-Level Performance

Collapse the 5-class problem into binary: **Any DR (grade ≥ 1) vs No DR (grade 0)**:

```
Referral Sensitivity = TP_refer / (TP_refer + FN_refer)
Referral Specificity = TN_refer / (TN_refer + FP_refer)
```

This is the most clinically important metric for a population screening tool. **Target: Sensitivity ≥ 90%, Specificity ≥ 80%** (engineering targets for the prototype — not clinical thresholds).

### 7. AUC-ROC (optional)

For binary referral decisions, compute the area under the ROC curve using the predicted probability for "Any DR":

```matlab
[~,~,~,AUC] = perfcurve(gtBinary, predScores, 1);
```

---

## Evaluation on IDRiD (Lesion Segmentation)

Once the segmentation model is trained, evaluate using:

### Dice Score (per lesion type)

```
Dice = 2 × |Pred ∩ GT| / (|Pred| + |GT|)
```

Ranges 0–1 (1 = perfect overlap). Report separately for MA, exudates, hemorrhages.

### AUROC (lesion-level sensitivity-specificity)

Vary the probability threshold and compute the ROC curve at the pixel or lesion level.

---

## Critical Evaluation Rules

### Rule 1: Test set isolation
The test set is used **exactly once** for final evaluation. Never:
- Tune hyperparameters on the test set
- Select the model checkpoint based on test set performance
- Report multiple runs on the test set

### Rule 2: No cherry-picking
Report all test images, including failures. Do not exclude difficult cases.

### Rule 3: Reproducibility
Document the exact random seed, train/val/test split indices, model checkpoint epoch, and MATLAB version used.

### Rule 4: Distinguish train/val from test metrics
Always clearly label which metric comes from which split:
- ❌ "Accuracy: 85%" (which split?)
- ✅ "Test set accuracy: 85% (N=732)"

### Rule 5: No clinical performance claims without formal study
Do not state sensitivity/specificity values as clinically validated without a peer-reviewed study. For SIH, report them as **"prototype evaluation results"**.

---

## Reporting Template

When real evaluation results are available, report:

```
RetinaGuard DR Classifier — Evaluation Results
================================================
Dataset:   APTOS 2019
Test set:  N = [number] (held-out, never used in training)
Model:     [architecture] fine-tuned on APTOS 2019 training set
MATLAB:    [version]
Date:      [date]

Overall Accuracy:        XX.X%

Per-Grade Results:
  Grade 0 (No DR):      Sens XX.X%  Spec XX.X%  F1 X.XX  N=XXX
  Grade 1 (Mild NPDR):  Sens XX.X%  Spec XX.X%  F1 X.XX  N=XXX
  Grade 2 (Mod NPDR):   Sens XX.X%  Spec XX.X%  F1 X.XX  N=XXX
  Grade 3 (Sev NPDR):   Sens XX.X%  Spec XX.X%  F1 X.XX  N=XXX
  Grade 4 (PDR):        Sens XX.X%  Spec XX.X%  F1 X.XX  N=XXX

Referral Performance (Grade ≥ 1):
  Sensitivity:  XX.X%
  Specificity:  XX.X%
  AUC:          X.XXX

NOTE: These are prototype evaluation results, not clinically validated metrics.
```

---

## Why Sensitivity Matters More Than Accuracy for Screening

In a population with 30% DR prevalence, a naive classifier that predicts "No DR" for everyone achieves 70% accuracy — but misses every DR case. For screening, **missing a DR case (false negative)** is far more harmful than an unnecessary referral (false positive).

Therefore:
- **Optimise for sensitivity first** (catch most DR cases)
- **Maintain acceptable specificity** (avoid overwhelming referral systems)
- **Never use accuracy alone** to evaluate a screening system

---

## Running the Test Suite

The automated test suite validates all pipeline functions with synthetic images:

```matlab
run('runAllTests.m')
```

Expected output (11 tests):
```
[PASS] Config loaded
[PASS] Synthetic fundus generation
[PASS] Quality assessment
[PASS] Ungradable detection
[PASS] Image enhancement
[PASS] Structure detection
[PASS] Lesion detection
[PASS] DR grading
[PASS] Full pipeline
[PASS] Demo cases
[PASS] Report generation

Results: 11/11 passed | 0 failed
ALL TESTS PASSED
```
