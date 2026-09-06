"""
RetinaGuard — EXP-002 Comprehensive Comparative Evaluation
===========================================================
Runs identical evaluation over:
  - Baseline
  - EXP-001
  - EXP-002-A
  - EXP-002-B
  - EXP-002-C
on the held-out validation set (N=733).
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

def evaluate_all():
    with open(SCRIPT_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    from dataset.aptos_loader import APTOSLoader
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    from models.efficientnet_dr import build_model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

    prep = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    _, val_ds = loader.get_splits(prep)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cpu")

    models_to_eval = [
        ("Baseline", PROJECT_ROOT / "models" / "baseline_model" / "best_model.pt"),
        ("EXP-001", PROJECT_ROOT / "models" / "experiments" / "EXP-001" / "best_model.pt"),
        ("EXP-002-A", PROJECT_ROOT / "models" / "experiments" / "EXP-002-A" / "best_model.pt"),
        ("EXP-002-B", PROJECT_ROOT / "models" / "experiments" / "EXP-002-B" / "best_model.pt"),
        ("EXP-002-C", PROJECT_ROOT / "models" / "experiments" / "EXP-002-C" / "best_model.pt"),
    ]

    results = {}

    for name, path in models_to_eval:
        if not path.exists():
            print(f"Skipping {name} (checkpoint not found at {path})")
            continue

        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()

        all_probs, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.numpy())
                all_preds.append(logits.argmax(dim=1).numpy())
                all_targets.append(lbls.numpy())

        probs = np.concatenate(all_probs, axis=0)
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        acc = accuracy_score(targets, preds)
        qwk = cohen_kappa_score(targets, preds, weights="quadratic", labels=list(range(5)))
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        prec = precision_score(targets, preds, average=None, zero_division=0)
        rec = recall_score(targets, preds, average=None, zero_division=0)
        f1s = f1_score(targets, preds, average=None, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=list(range(5)))

        t_ref = (targets >= 2).astype(int)
        p_ref = (preds >= 2).astype(int)
        tp = np.sum((t_ref == 1) & (p_ref == 1))
        fn = np.sum((t_ref == 1) & (p_ref == 0))
        fp = np.sum((t_ref == 0) & (p_ref == 1))
        tn = np.sum((t_ref == 0) & (p_ref == 0))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        g4_idx = np.where(targets == 4)[0]
        g4_pred_counts = np.bincount(preds[g4_idx], minlength=5)
        g4_mean_probs = probs[g4_idx].mean(axis=0)

        results[name] = {
            "accuracy": acc, "qwk": qwk, "macro_f1": macro_f1,
            "precision": prec, "recall": rec, "f1": f1s,
            "sens": sens, "spec": spec, "cm": cm,
            "g4_pred_counts": g4_pred_counts,
            "g4_mean_probs": g4_mean_probs
        }

    print("\n" + "=" * 105)
    print("  EXP-002 SUITE: COMPREHENSIVE PERFORMANCE MATRIX")
    print("=" * 105)
    header = f"{'Experiment':<12} | {'QWK':<6} | {'Acc (%)':<7} | {'Mac F1':<6} | {'G4 Rec':<6} | {'G4 F1':<6} | {'Ref Sens':<8} | {'Ref Spec':<8}"
    print(header)
    print("-" * 105)
    for name, r in results.items():
        print(f"{name:<12} | {r['qwk']:.4f} | {r['accuracy']*100:6.2f}% | {r['macro_f1']:.4f} | {r['recall'][4]:.4f} | {r['f1'][4]:.4f} | {r['sens']*100:6.2f}%  | {r['spec']*100:6.2f}%")
    print("=" * 105)

    print("\n" + "=" * 90)
    print("  DETAILED TRUE GRADE 4 BREAKDOWN (N = 59)")
    print("=" * 90)
    g4_hdr = f"{'Experiment':<12} | {'Pred 0':<6} | {'Pred 1':<6} | {'Pred 2':<6} | {'Pred 3':<6} | {'Pred 4':<6} | {'P(G2)':<6} | {'P(G3)':<6} | {'P(G4)':<6}"
    print(g4_hdr)
    print("-" * 90)
    for name, r in results.items():
        cnts = r["g4_pred_counts"]
        m_p = r["g4_mean_probs"]
        print(f"{name:<12} | {cnts[0]:<6} | {cnts[1]:<6} | {cnts[2]:<6} | {cnts[3]:<6} | {cnts[4]:<6} | {m_p[2]:.4f} | {m_p[3]:.4f} | {m_p[4]:.4f}")
    print("=" * 90)

    # Save to JSON
    json_out = {}
    for name, r in results.items():
        json_out[name] = {
            "qwk": float(r["qwk"]), "accuracy": float(r["accuracy"]), "macro_f1": float(r["macro_f1"]),
            "sens": float(r["sens"]), "spec": float(r["spec"]),
            "g4_recall": float(r["recall"][4]), "g4_f1": float(r["f1"][4]), "g4_precision": float(r["precision"][4]),
            "g4_pred_counts": [int(x) for x in r["g4_pred_counts"]],
            "g4_mean_probs": [float(x) for x in r["g4_mean_probs"]],
            "confusion_matrix": r["cm"].tolist()
        }
    with open(PROJECT_ROOT / "results" / "exp002_comparison.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved complete results to: {PROJECT_ROOT / 'results' / 'exp002_comparison.json'}")

if __name__ == "__main__":
    evaluate_all()
