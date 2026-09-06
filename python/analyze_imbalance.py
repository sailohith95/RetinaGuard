"""
Phase 2 & Phase 6 Analysis:
Class Imbalance & Grade 4 Deep Dive
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

def analyze():
    print("=" * 70)
    print("  PHASE 2 & 6: CLASS IMBALANCE & GRADE 4 ERROR ANALYSIS")
    print("=" * 70)

    with open(SCRIPT_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    from dataset.aptos_loader import APTOSLoader
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    from models.efficientnet_dr import build_model

    prep = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    train_ds, val_ds = loader.get_splits(prep)

    train_labels = train_ds.get_labels()
    val_labels = val_ds.get_labels()
    train_counts = np.bincount(train_labels, minlength=5)
    val_counts = np.bincount(val_labels, minlength=5)

    print("\nExact Class Distribution:")
    print(f"{'Class':<22} {'Train Count':<12} {'Train %':<10} {'Val Count':<10} {'Val %':<10} {'Class Weight':<12}")
    weights = loader.get_class_weights(train_labels)
    for i in range(5):
        t_pct = 100 * train_counts[i] / len(train_labels)
        v_pct = 100 * val_counts[i] / len(val_labels)
        print(f"{CLASS_NAMES[i]:<22} {train_counts[i]:<12} {t_pct:<10.2f} {val_counts[i]:<10} {v_pct:<10.2f} {weights[i]:<12.4f}")

    # Load baseline model to analyze Grade 4 predictions specifically
    device = torch.device("cpu")
    model = build_model(cfg)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "baseline_model" / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    all_logits = []
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, lbls in val_loader:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_logits.append(logits.numpy())
            all_probs.append(probs.numpy())
            all_preds.append(logits.argmax(dim=1).numpy())
            all_targets.append(lbls.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Filter to True Grade 4 samples
    g4_idx = np.where(all_targets == 4)[0]
    g4_preds = all_preds[g4_idx]
    g4_probs = all_probs[g4_idx]
    g4_pred_counts = np.bincount(g4_preds, minlength=5)

    print("\nGrade 4 Detailed Diagnosis:")
    print(f"Total True Grade 4 samples in Validation Set: {len(g4_idx)}")
    print("Where did the model misclassify Grade 4 samples?")
    for c in range(5):
        cnt = g4_pred_counts[c]
        pct = 100 * cnt / len(g4_idx)
        print(f"  Predicted as {CLASS_NAMES[c]:<22}: {cnt:2d} ({pct:5.1f}%)")

    # Mean predicted probability across all 5 classes for True Grade 4 samples
    mean_g4_probs = g4_probs.mean(axis=0)
    print("\nMean Predicted Probability Vector for True Grade 4 samples:")
    for c in range(5):
        print(f"  P({CLASS_NAMES[c]:<22}): {mean_g4_probs[c]:.4f}")

    # Save detailed error breakdown plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Overall Class distribution
    x = np.arange(5)
    width = 0.35
    axes[0].bar(x - width/2, train_counts, width, label='Train', color='#1b5ca8')
    axes[0].bar(x + width/2, val_counts, width, label='Val', color='#c47d00')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=20)
    axes[0].set_title('Dataset Distribution per Class')
    axes[0].set_ylabel('Number of Samples')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: True Grade 4 Predictions
    bars = axes[1].bar(CLASS_NAMES, g4_pred_counts, color=['#e66101', '#fdb863', '#b2abd2', '#5e3c99', '#2d7a2d'])
    axes[1].set_title('Distribution of Model Predictions for True Grade 4 Images (N=59)')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=20)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{int(h)} ({100*h/len(g4_idx):.1f}%)", ha='center', va='bottom', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    results_dir = PROJECT_ROOT / "results"
    out_img = results_dir / "grade4_error_analysis.png"
    fig.savefig(out_img, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Grade 4 error analysis plot saved to: {out_img}")

    # Output findings to a JSON file
    summary = {
        "train_counts": train_counts.tolist(),
        "val_counts": val_counts.tolist(),
        "weights": weights.tolist(),
        "grade4_val_total": int(len(g4_idx)),
        "grade4_predictions": g4_pred_counts.tolist(),
        "grade4_mean_predicted_probabilities": [float(p) for p in mean_g4_probs]
    }
    with open(results_dir / "grade4_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Grade 4 analysis data saved to: {results_dir / 'grade4_analysis.json'}")

if __name__ == "__main__":
    analyze()
