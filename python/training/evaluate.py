"""
RetinaGuard — Evaluation Module
=================================
Generates comprehensive evaluation metrics after training.
Can also be run standalone: python training/evaluate.py

Computes:
  - Accuracy, Precision, Recall, F1 (per-class + macro)
  - Quadratic Weighted Kappa (QWK)
  - Confusion matrix (with visualization)
  - Referable DR performance (grade >= 2)
  - Classification report

All metrics are computed from REAL model predictions — never invented.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
PYTHON_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


def evaluate_model(
    model: nn.Module,
    val_loader,
    device: torch.device,
    results_dir: Path,
    model_dir: Path,
) -> dict:
    """
    Run full evaluation and save all results.
    Returns metrics dict.
    """
    results_dir = Path(results_dir)
    model_dir = Path(model_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running evaluation on validation set…")

    # Collect predictions
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ── Metrics ────────────────────────────────────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, cohen_kappa_score,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                  labels=list(range(5))))

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_recall    = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_f1        = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                        zero_division=0)

    # Referable DR: grade >= 2 → refer
    REFERRAL_THRESHOLD = 2
    y_true_ref = (y_true >= REFERRAL_THRESHOLD).astype(int)
    y_pred_ref = (y_pred >= REFERRAL_THRESHOLD).astype(int)
    tp = int(np.sum((y_true_ref == 1) & (y_pred_ref == 1)))
    fn = int(np.sum((y_true_ref == 1) & (y_pred_ref == 0)))
    fp = int(np.sum((y_true_ref == 0) & (y_pred_ref == 1)))
    tn = int(np.sum((y_true_ref == 0) & (y_pred_ref == 0)))
    ref_sensitivity = tp / max(tp + fn, 1)
    ref_specificity = tn / max(tn + fp, 1)

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "qwk": round(qwk, 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(per_class_precision[i], 4),
                "recall": round(per_class_recall[i], 4),
                "f1": round(per_class_f1[i], 4),
            }
            for i in range(5)
        },
        "referable_dr_threshold": REFERRAL_THRESHOLD,
        "referable_sensitivity": round(ref_sensitivity, 4),
        "referable_specificity": round(ref_specificity, 4),
        "n_val_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
    }

    # -- Print summary ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RetinaGuard -- Evaluation Results")
    print("=" * 65)
    print(f"  Validation samples : {len(y_true)}")
    print(f"  Accuracy           : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Macro F1           : {macro_f1:.4f}")
    print(f"  QWK                : {qwk:.4f}")
    print()
    print("  Per-class results:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    Grade {i} ({name:<22}): "
              f"P={per_class_precision[i]:.3f}  R={per_class_recall[i]:.3f}  "
              f"F1={per_class_f1[i]:.3f}")
    print()
    print(f"  Referable DR (>= Grade {REFERRAL_THRESHOLD}):")
    print(f"    Sensitivity: {ref_sensitivity:.4f}  ({ref_sensitivity*100:.1f}%)")
    print(f"    Specificity: {ref_specificity:.4f}  ({ref_specificity*100:.1f}%)")
    print("=" * 65 + "\n")
    print("Classification Report:")
    print(clf_report)

    # ── Save results ───────────────────────────────────────────────────────────
    with open(results_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write("RetinaGuard — DR Classifier Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"QWK:      {qwk:.4f}\n\n")
        f.write(clf_report)

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    _plot_confusion_matrix(cm, results_dir)

    logger.info("Evaluation complete. QWK=%.4f, Accuracy=%.4f", qwk, accuracy)
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, results_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        # Normalize for display
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {i}" for i in range(5)],
            yticklabels=[f"True {i}\n{CLASS_NAMES[i]}" for i in range(5)],
            linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title("RetinaGuard — Confusion Matrix (val set)\n"
                     "Counts shown; color = row-normalized fraction", fontsize=10)
        ax.set_xlabel("Predicted Grade")
        ax.set_ylabel("True Grade")
        plt.tight_layout()
        fig.savefig(results_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", results_dir / "confusion_matrix.png")
    except Exception as e:
        logger.warning("Could not generate confusion matrix plot: %s", e)


# ==============================================================================
#  Standalone entry point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate RetinaGuard DR classifier")
    parser.add_argument("--config", default=str(PYTHON_DIR / "config" / "config.yaml"))
    parser.add_argument("--model", default=None, help="Override model path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.efficientnet_dr import build_model
    model = build_model(cfg)
    model_path = args.model or (PROJECT_ROOT / cfg["paths"]["best_model"])
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}. Train first.")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)

    from dataset.aptos_loader import APTOSLoader
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    preprocessor = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    _, val_ds = loader.get_splits(preprocessor)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0
    )

    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    evaluate_model(model, val_loader, device, results_dir, model_dir)
