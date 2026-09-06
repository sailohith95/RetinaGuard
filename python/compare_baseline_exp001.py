"""
RetinaGuard — Rigorous Baseline vs EXP-001 Comparison Script
============================================================
Performs exact head-to-head comparison on the held-out validation set.
Outputs all requested metrics and specific Grade 4 probability breakdowns.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

def run_comparison():
    print("=" * 80)
    print("  RIGOROUS BASELINE vs EXP-001 COMPARISON (HELD-OUT VALIDATION SET: N=733)")
    print("=" * 80)

    with open(SCRIPT_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    from dataset.aptos_loader import APTOSLoader
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    from models.efficientnet_dr import build_model

    prep = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    _, val_ds = loader.get_splits(prep)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cpu")

    # 1. Evaluate Baseline Model
    baseline_path = PROJECT_ROOT / "models" / "baseline_model" / "best_model.pt"
    m_base = build_model(cfg)
    m_base.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
    m_base.eval()

    # 2. Evaluate EXP-001 Model
    exp001_path = PROJECT_ROOT / "models" / "experiments" / "EXP-001" / "best_model.pt"
    if not exp001_path.exists():
        print(f"Error: EXP-001 model not found at {exp001_path}. Waiting for training completion...")
        return False

    m_exp = build_model(cfg)
    m_exp.load_state_dict(torch.load(exp001_path, map_location=device, weights_only=True))
    m_exp.eval()

    # Run inference for both
    def get_preds(model):
        all_logits, all_probs, all_preds, all_targets = [], [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                all_logits.append(logits.numpy())
                all_probs.append(probs.numpy())
                all_preds.append(logits.argmax(dim=1).numpy())
                all_targets.append(lbls.numpy())
        return (
            np.concatenate(all_logits, axis=0),
            np.concatenate(all_probs, axis=0),
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_targets, axis=0)
        )

    b_logits, b_probs, b_preds, targets = get_preds(m_base)
    e_logits, e_probs, e_preds, _ = get_preds(m_exp)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

    # Compute comprehensive metrics
    def compute_all_metrics(preds, probs):
        acc = accuracy_score(targets, preds)
        qwk = cohen_kappa_score(targets, preds, weights="quadratic", labels=list(range(5)))
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        prec = precision_score(targets, preds, average=None, zero_division=0)
        rec = recall_score(targets, preds, average=None, zero_division=0)
        f1s = f1_score(targets, preds, average=None, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=list(range(5)))

        # Referable DR (>=2)
        t_ref = (targets >= 2).astype(int)
        p_ref = (preds >= 2).astype(int)
        tp = np.sum((t_ref == 1) & (p_ref == 1))
        fn = np.sum((t_ref == 1) & (p_ref == 0))
        fp = np.sum((t_ref == 0) & (p_ref == 1))
        tn = np.sum((t_ref == 0) & (p_ref == 0))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        return {
            "accuracy": acc, "qwk": qwk, "macro_f1": macro_f1,
            "precision": prec, "recall": rec, "f1": f1s,
            "cm": cm, "sens": sens, "spec": spec
        }

    m_b = compute_all_metrics(b_preds, b_probs)
    m_e = compute_all_metrics(e_preds, e_probs)

    print("\n" + "=" * 80)
    print(f"{'Metric':<30} {'Baseline':<12} {'EXP-001':<12} {'Change':<12} {'Better Model':<12}")
    print("-" * 80)

    def print_comparison_row(name, val_b, val_e, higher_is_better=True, is_pct=False):
        diff = val_e - val_b
        if abs(diff) < 1e-4:
            better = "Equal"
        elif (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
            better = "EXP-001"
        else:
            better = "Baseline"
        
        fmt = "{:.2f}%" if is_pct else "{:.4f}"
        diff_fmt = "{:+.2f}%" if is_pct else "{:+.4f}"
        b_str = fmt.format(val_b * 100 if is_pct else val_b)
        e_str = fmt.format(val_e * 100 if is_pct else val_e)
        d_str = diff_fmt.format(diff * 100 if is_pct else diff)
        print(f"{name:<30} {b_str:<12} {e_str:<12} {d_str:<12} {better:<12}")

    print_comparison_row("QWK (Primary Benchmark)", m_b["qwk"], m_e["qwk"])
    print_comparison_row("Macro F1", m_b["macro_f1"], m_e["macro_f1"])
    print_comparison_row("Overall Accuracy", m_b["accuracy"], m_e["accuracy"], is_pct=True)
    print_comparison_row("Referable Sensitivity (>=G2)", m_b["sens"], m_e["sens"], is_pct=True)
    print_comparison_row("Referable Specificity (<G2)", m_b["spec"], m_e["spec"], is_pct=True)
    print("-" * 80)
    for i, cname in enumerate(CLASS_NAMES):
        print_comparison_row(f"{cname} Precision", m_b["precision"][i], m_e["precision"][i])
        print_comparison_row(f"{cname} Recall", m_b["recall"][i], m_e["recall"][i])
        print_comparison_row(f"{cname} F1-Score", m_b["f1"][i], m_e["f1"][i])
        if i < 4:
            print("." * 80)

    # Confusion Matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES (True Rows, Pred Cols):")
    print("Baseline:")
    print("        Pred 0  Pred 1  Pred 2  Pred 3  Pred 4   Total")
    for i, row in enumerate(m_b["cm"]):
        print(f"True {i}: {row[0]:6d}  {row[1]:6d}  {row[2]:6d}  {row[3]:6d}  {row[4]:6d}   {sum(row):5d}")

    print("\nEXP-001:")
    print("        Pred 0  Pred 1  Pred 2  Pred 3  Pred 4   Total")
    for i, row in enumerate(m_e["cm"]):
        print(f"True {i}: {row[0]:6d}  {row[1]:6d}  {row[2]:6d}  {row[3]:6d}  {row[4]:6d}   {sum(row):5d}")

    # Specific True Grade 4 Analysis
    g4_indices = np.where(targets == 4)[0]
    total_g4 = len(g4_indices)
    print("\n" + "=" * 80)
    print(f"SPECIFIC TRUE GRADE 4 ANALYSIS (Total Samples = {total_g4}):")
    print("=" * 80)

    b_g4_preds = b_preds[g4_indices]
    e_g4_preds = e_preds[g4_indices]

    b_g4_pred_counts = np.bincount(b_g4_preds, minlength=5)
    e_g4_pred_counts = np.bincount(e_g4_preds, minlength=5)

    b_g4_mean_probs = b_probs[g4_indices].mean(axis=0)
    e_g4_mean_probs = e_probs[g4_indices].mean(axis=0)

    print(f"{'Metric on True Grade 4':<35} {'Baseline':<14} {'EXP-001':<14} {'Change':<14}")
    print("-" * 80)
    print(f"{'Predicted as Grade 4 (Correct)':<35} {b_g4_pred_counts[4]:<14} {e_g4_pred_counts[4]:<14} {e_g4_pred_counts[4] - b_g4_pred_counts[4]:+d}")
    print(f"{'Predicted as Grade 3 (Severe)':<35} {b_g4_pred_counts[3]:<14} {e_g4_pred_counts[3]:<14} {e_g4_pred_counts[3] - b_g4_pred_counts[3]:+d}")
    print(f"{'Predicted as Grade 2 (Moderate)':<35} {b_g4_pred_counts[2]:<14} {e_g4_pred_counts[2]:<14} {e_g4_pred_counts[2] - b_g4_pred_counts[2]:+d}")
    print(f"{'Predicted as Grade 1 (Mild)':<35} {b_g4_pred_counts[1]:<14} {e_g4_pred_counts[1]:<14} {e_g4_pred_counts[1] - b_g4_pred_counts[1]:+d}")
    print(f"{'Predicted as Grade 0 (No DR)':<35} {b_g4_pred_counts[0]:<14} {e_g4_pred_counts[0]:<14} {e_g4_pred_counts[0] - b_g4_pred_counts[0]:+d}")
    print("-" * 80)
    print(f"{'Mean P(Grade 4 Proliferative)':<35} {b_g4_mean_probs[4]:.4f}         {e_g4_mean_probs[4]:.4f}         {e_g4_mean_probs[4] - b_g4_mean_probs[4]:+.4f}")
    print(f"{'Mean P(Grade 3 Severe)':<35} {b_g4_mean_probs[3]:.4f}         {e_g4_mean_probs[3]:.4f}         {e_g4_mean_probs[3] - b_g4_mean_probs[3]:+.4f}")
    print(f"{'Mean P(Grade 2 Moderate)':<35} {b_g4_mean_probs[2]:.4f}         {e_g4_mean_probs[2]:.4f}         {e_g4_mean_probs[2] - b_g4_mean_probs[2]:+.4f}")

    # Output to JSON for record
    res_data = {
        "baseline": {
            "qwk": float(m_b["qwk"]), "macro_f1": float(m_b["macro_f1"]), "accuracy": float(m_b["accuracy"]),
            "sens": float(m_b["sens"]), "spec": float(m_b["spec"]),
            "g4_recall": float(m_b["recall"][4]), "g4_f1": float(m_b["f1"][4]), "g4_precision": float(m_b["precision"][4]),
            "g4_pred_as_4": int(b_g4_pred_counts[4]), "g4_pred_as_3": int(b_g4_pred_counts[3]), "g4_pred_as_2": int(b_g4_pred_counts[2]),
            "mean_p_g4": float(b_g4_mean_probs[4]), "mean_p_g3": float(b_g4_mean_probs[3]), "mean_p_g2": float(b_g4_mean_probs[2])
        },
        "exp001": {
            "qwk": float(m_e["qwk"]), "macro_f1": float(m_e["macro_f1"]), "accuracy": float(m_e["accuracy"]),
            "sens": float(m_e["sens"]), "spec": float(m_e["spec"]),
            "g4_recall": float(m_e["recall"][4]), "g4_f1": float(m_e["f1"][4]), "g4_precision": float(m_e["precision"][4]),
            "g4_pred_as_4": int(e_g4_pred_counts[4]), "g4_pred_as_3": int(e_g4_pred_counts[3]), "g4_pred_as_2": int(e_g4_pred_counts[2]),
            "mean_p_g4": float(e_g4_mean_probs[4]), "mean_p_g3": float(e_g4_mean_probs[3]), "mean_p_g2": float(e_g4_mean_probs[2])
        }
    }
    with open(PROJECT_ROOT / "results" / "baseline_vs_exp001.json", "w", encoding="utf-8") as f:
        json.dump(res_data, f, indent=2)

    return True

if __name__ == "__main__":
    run_comparison()
