"""
RetinaGuard — EXP-003 Stage-2 Backbone Fine-Tuning Suite
=========================================================
Fine-tunes the upper EfficientNet-B0 convolutional blocks starting
from the candidate best EXP-001 checkpoint (QWK=0.7342).

Variants:
  - EXP-003-A: Top 2 blocks unfrozen (MBConv Stage 7 + 1x1 Conv)
  - EXP-003-B: Top 3 blocks unfrozen (MBConv Stages 6 & 7 + 1x1 Conv)

Uses:
  - Differential learning rates:
      Backbone: 2e-5
      Classifier head: 1e-4
  - Focal loss (gamma=2.0) with standard unweighted sampler (matching EXP-001)
  - Validation-based checkpointing (tracking QWK) & early stopping
  - Exact held-out validation set (N=733)
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.aptos_loader import APTOSLoader
from preprocessing.retinal_preprocessor import RetinalPreprocessor
from models.efficientnet_dr import build_model, DRClassifier
from training.train import set_seed, get_device, train_epoch, val_epoch, compute_qwk, compute_macro_f1
from training.evaluate import evaluate_model
from training.losses import build_loss_function
from training.experiments import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinaguard.exp003")

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


def fine_tune_variant(
    exp_id: str,
    unfreeze_last_n_blocks: int,
    epochs: int = 3,
    lr_backbone: float = 2e-5,
    lr_head: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    seed: int = 42
) -> Dict:
    set_seed(seed)
    device = get_device()

    logger.info("=" * 75)
    logger.info("STARTING FINE-TUNING EXPERIMENT: %s", exp_id)
    logger.info("Unfreeze depth: Top %d blocks | LR Backbone: %.1e | LR Head: %.1e | Epochs: %d",
                unfreeze_last_n_blocks, lr_backbone, lr_head, epochs)
    logger.info("=" * 75)

    # 1. Config & Data Setup
    with open(PYTHON_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    prep = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    train_ds, val_ds = loader.get_splits(prep)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    train_labels = train_ds.get_labels()
    train_counts = np.bincount(train_labels, minlength=5)
    class_weights = loader.get_class_weights(train_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 2. Criterion: Identical to EXP-001 (Focal Loss, gamma=2.0)
    criterion = build_loss_function(
        loss_name="focal",
        class_weights=weight_tensor,
        train_counts=train_counts.tolist(),
        gamma=2.0
    )

    # 3. Model Architecture & Initialization from EXP-001 Checkpoint
    model = build_model(cfg).to(device)
    exp001_ckpt = PROJECT_ROOT / "models" / "experiments" / "EXP-001" / "best_model.pt"
    assert exp001_ckpt.exists(), f"EXP-001 checkpoint missing at {exp001_ckpt}"

    logger.info("Loading baseline weights from EXP-001: %s", exp001_ckpt)
    model.load_state_dict(torch.load(exp001_ckpt, map_location=device, weights_only=True))

    # 4. Unfreeze Chosen Blocks
    model.freeze_backbone()
    model.unfreeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    # Ensure classifier head is trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    param_stats = model.get_trainable_params()
    logger.info("Trainable parameters: %d (%.1f%%) | Frozen parameters: %d",
                param_stats["trainable"], 100 * param_stats["trainable"] / param_stats["total"],
                param_stats["frozen"])

    # 5. Differential Learning Rate Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head}
        ],
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 6. Training Directory & Checkpoint
    exp_dir = PROJECT_ROOT / "models" / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_ckpt = exp_dir / "best_model.pt"

    # Pre-fine-tuning evaluation to verify baseline state
    logger.info("Verifying initial pre-fine-tuning baseline performance on validation set...")
    init_vloss, init_vacc, init_preds, init_labels = val_epoch(model, val_loader, criterion, device)
    init_qwk = compute_qwk(init_labels, init_preds)
    init_f1 = compute_macro_f1(init_labels, init_preds)
    logger.info("  Initial State: Val Loss=%.4f | Val Acc=%.3f | Val QWK=%.4f | Val F1=%.4f",
                init_vloss, init_vacc, init_qwk, init_f1)

    best_qwk = init_qwk
    best_epoch = 0
    # Save initial state as fallback best checkpoint
    torch.save(model.state_dict(), exp_ckpt)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [init_vloss], "val_acc": [init_vacc],
        "val_qwk": [init_qwk], "val_f1": [init_f1]
    }

    # 7. Fine-Tuning Loop with Overfitting Monitoring
    patience = 2
    no_improve_epochs = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_preds, v_labels = val_epoch(model, val_loader, criterion, device)
        qwk = compute_qwk(v_labels, v_preds)
        f1 = compute_macro_f1(v_labels, v_preds)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_qwk"].append(qwk)
        history["val_f1"].append(f1)

        elapsed = time.time() - t0
        logger.info(
            "  Ep %d/%d | TrainLoss=%.4f TrainAcc=%.3f | ValLoss=%.4f ValAcc=%.3f QWK=%.4f F1=%.4f | %.1fs",
            ep, epochs, t_loss, t_acc, v_loss, v_acc, qwk, f1, elapsed
        )

        if qwk > best_qwk:
            best_qwk = qwk
            best_epoch = ep
            torch.save(model.state_dict(), exp_ckpt)
            no_improve_epochs = 0
            logger.info("    [OK] New best checkpoint saved for %s (Epoch %d, QWK=%.4f)", exp_id, ep, best_qwk)
        else:
            no_improve_epochs += 1
            logger.info("    [-] QWK did not improve (current: %.4f, best: %.4f). Patience count: %d/%d",
                        qwk, best_qwk, no_improve_epochs, patience)

        if no_improve_epochs >= patience:
            logger.info("Early stopping triggered at epoch %d.", ep)
            break

    # 8. Evaluate Best Checkpoint
    logger.info("Loading best checkpoint (Epoch %d, QWK=%.4f) for detailed evaluation...", best_epoch, best_qwk)
    model.load_state_dict(torch.load(exp_ckpt, map_location=device, weights_only=True))
    metrics = evaluate_model(model, val_loader, device, exp_dir, exp_dir)

    exp_result = {
        "exp_id": exp_id,
        "model": "EfficientNet-B0",
        "loss": "focal",
        "sampler": "Standard",
        "stage1_epochs": 3,
        "stage2_epochs": epochs,
        "unfreeze_depth": f"Top {unfreeze_last_n_blocks} blocks",
        "lr_stage1": 1e-3,
        "lr_stage2": lr_backbone,
        "lr_head": lr_head,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "checkpoint_path": str(exp_ckpt),
        "history": history,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    tracker = ExperimentTracker(PROJECT_ROOT / "results" / "experiments_log.json")
    tracker.record(exp_result)

    return exp_result


def run_comparative_evaluation():
    logger.info("\n" + "=" * 80)
    logger.info("  RUNNING UNIFIED EXP-001 vs EXP-003 COMPARATIVE EVALUATION")
    logger.info("=" * 80)

    with open(PYTHON_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    prep = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    _, val_ds = loader.get_splits(prep)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cpu")

    models_to_eval = [
        ("EXP-001", PROJECT_ROOT / "models" / "experiments" / "EXP-001" / "best_model.pt"),
        ("EXP-003-A", PROJECT_ROOT / "models" / "experiments" / "EXP-003-A" / "best_model.pt"),
        ("EXP-003-B", PROJECT_ROOT / "models" / "experiments" / "EXP-003-B" / "best_model.pt"),
    ]

    results = {}

    for name, path in models_to_eval:
        if not path.exists():
            logger.warning("Checkpoint for %s not found at %s", name, path)
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

    # Print Main Comparison Table
    print("\n" + "=" * 115)
    print("  EXP-003 COMPARATIVE PERFORMANCE MATRIX (HELD-OUT VALIDATION SET: N=733)")
    print("=" * 115)
    header = f"{'Model':<12} | {'QWK':<7} | {'Acc (%)':<7} | {'Mac F1':<7} | {'G2 F1':<6} | {'G3 F1':<6} | {'G4 F1':<6} | {'G4 Rec':<6} | {'Ref Sens':<8} | {'Ref Spec':<8}"
    print(header)
    print("-" * 115)
    for name, r in results.items():
        print(f"{name:<12} | {r['qwk']:.4f}  | {r['accuracy']*100:6.2f}% | {r['macro_f1']:.4f}  | {r['f1'][2]:.4f} | {r['f1'][3]:.4f} | {r['f1'][4]:.4f} | {r['recall'][4]:.4f} | {r['sens']*100:6.2f}%  | {r['spec']*100:6.2f}%")
    print("=" * 115)

    # Print Delta from EXP-001
    if "EXP-001" in results:
        base = results["EXP-001"]
        print("\n" + "=" * 105)
        print("  ABSOLUTE CHANGE RELATIVE TO EXP-001 BASELINE (Δ = Variant - EXP-001)")
        print("=" * 105)
        delta_hdr = f"{'Model':<12} | {'Δ QWK':<9} | {'Δ Acc':<9} | {'Δ Mac F1':<9} | {'Δ G4 Rec':<9} | {'Δ G4 F1':<9} | {'Δ Ref Sens':<11} | {'Δ Ref Spec':<11}"
        print(delta_hdr)
        print("-" * 105)
        for name in ["EXP-003-A", "EXP-003-B"]:
            if name in results:
                r = results[name]
                d_qwk = r["qwk"] - base["qwk"]
                d_acc = (r["accuracy"] - base["accuracy"]) * 100
                d_f1 = r["macro_f1"] - base["macro_f1"]
                d_g4rec = (r["recall"][4] - base["recall"][4]) * 100
                d_g4f1 = r["f1"][4] - base["f1"][4]
                d_sens = (r["sens"] - base["sens"]) * 100
                d_spec = (r["spec"] - base["spec"]) * 100
                print(f"{name:<12} | {d_qwk:+7.4f}   | {d_acc:+6.2f}%   | {d_f1:+7.4f}   | {d_g4rec:+6.2f}%   | {d_g4f1:+7.4f}   | {d_sens:+8.2f}%   | {d_spec:+8.2f}%")
        print("=" * 105)

    # Print Grade 4 Analysis
    print("\n" + "=" * 90)
    print("  TRUE GRADE 4 (PDR) PREDICTION SPREAD (N = 59)")
    print("=" * 90)
    g4_hdr = f"{'Model':<12} | {'Pred 0':<6} | {'Pred 1':<6} | {'Pred 2':<6} | {'Pred 3':<6} | {'Pred 4':<6} | {'P(G4)':<6}"
    print(g4_hdr)
    print("-" * 90)
    for name, r in results.items():
        cnts = r["g4_pred_counts"]
        m_p = r["g4_mean_probs"]
        print(f"{name:<12} | {cnts[0]:<6} | {cnts[1]:<6} | {cnts[2]:<6} | {cnts[3]:<6} | {cnts[4]:<6} | {m_p[4]:.4f}")
    print("=" * 90)

    # Save to JSON
    json_out = {}
    for name, r in results.items():
        json_out[name] = {
            "qwk": float(r["qwk"]), "accuracy": float(r["accuracy"]), "macro_f1": float(r["macro_f1"]),
            "sens": float(r["sens"]), "spec": float(r["spec"]),
            "g2_f1": float(r["f1"][2]), "g3_f1": float(r["f1"][3]),
            "g4_recall": float(r["recall"][4]), "g4_f1": float(r["f1"][4]), "g4_precision": float(r["precision"][4]),
            "g4_pred_counts": [int(x) for x in r["g4_pred_counts"]],
            "g4_mean_probs": [float(x) for x in r["g4_mean_probs"]],
            "confusion_matrix": r["cm"].tolist()
        }
    comp_file = PROJECT_ROOT / "results" / "exp003_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved complete comparative evaluation to: {comp_file}")


def main():
    print("=" * 80)
    print("  LAUNCHING RETINAGUARD EXP-003 STAGE-2 BACKBONE FINE-TUNING SUITE")
    print("=" * 80)

    # Run EXP-003-A (Top 2 blocks unfrozen)
    fine_tune_variant(
        exp_id="EXP-003-A",
        unfreeze_last_n_blocks=2,
        epochs=3,
        lr_backbone=2e-5,
        lr_head=1e-4,
        weight_decay=1e-4,
        batch_size=32,
        seed=42
    )

    # Run EXP-003-B (Top 3 blocks unfrozen)
    fine_tune_variant(
        exp_id="EXP-003-B",
        unfreeze_last_n_blocks=3,
        epochs=3,
        lr_backbone=2e-5,
        lr_head=1e-4,
        weight_decay=1e-4,
        batch_size=32,
        seed=42
    )

    # Run comparative evaluation
    run_comparative_evaluation()


if __name__ == "__main__":
    main()
