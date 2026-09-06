"""
RetinaGuard — Training Pipeline
=================================
Full training loop for EfficientNet-B0 DR classifier.

Usage:
    python training/train.py
    python training/train.py --config python/config/config.yaml
    python training/train.py --epochs 5 --batch_size 16   # quick test

Execution order:
  1. Load config
  2. Set random seeds
  3. Inspect dataset (Phase 2)
  4. Create datasets + dataloaders
  5. Build EfficientNet-B0 model
  6. Stage 1: Train head (frozen backbone)
  7. Stage 2: Fine-tune upper backbone layers
  8. Save best model + metadata
  9. Run final evaluation
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# ── Path setup so we can import python/* modules ──────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PYTHON_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retinaguard.train")


# ==============================================================================
#  Helpers
# ==============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("No GPU found — training on CPU. This will be slow.")
    return device


def make_optimizer(model: nn.Module, cfg: dict, stage: int) -> torch.optim.Optimizer:
    t_cfg = cfg["training"]
    lr = t_cfg["learning_rate"] if stage == 1 else t_cfg.get("finetune_lr", 1e-4)
    wd = float(t_cfg.get("weight_decay", 1e-4))
    opt_name = t_cfg.get("optimizer", "adam").lower()

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def make_scheduler(optimizer, cfg: dict, n_epochs: int):
    scheduler_name = cfg["training"].get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    return None


# ==============================================================================
#  Training / Validation loops
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                "  Batch %d/%d  loss=%.4f  acc=%.3f",
                batch_idx + 1, len(loader),
                total_loss / total, correct / total
            )

    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Quadratic Weighted Kappa — key metric for ordinal DR grading."""
    from sklearn.metrics import cohen_kappa_score
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                       labels=list(range(n_classes))))
    except Exception:
        return 0.0


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ==============================================================================
#  Main training function
# ==============================================================================

def train(cfg: dict, args: argparse.Namespace):
    seed = cfg["training"]["seed"]
    set_seed(seed)
    device = get_device()

    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    model_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    ckpt_dir = PROJECT_ROOT / cfg["paths"]["checkpoint_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 2: Dataset inspection ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2 — Dataset Inspection")
    logger.info("=" * 60)
    from dataset.aptos_loader import APTOSLoader
    loader = APTOSLoader(cfg)
    inspect_report = loader.inspect()

    # Save inspection report
    with open(results_dir / "dataset_inspection.json", "w") as f:
        json.dump(inspect_report, f, indent=2)

    # ── Phase 3-4: Preprocessing + split ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3-4 — Preprocessing & Splitting")
    logger.info("=" * 60)
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    preprocessor = RetinalPreprocessor(cfg)

    train_ds, val_ds = loader.get_splits(preprocessor)

    # ── Phase 5: Class weights ────────────────────────────────────────────────
    train_labels = train_ds.get_labels()
    class_weights = loader.get_class_weights(train_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Weighted cross-entropy loss
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # DataLoaders
    batch_size = int(args.batch_size or cfg["training"]["batch_size"])
    n_workers = int(cfg["training"].get("num_workers", 0))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=cfg["training"].get("pin_memory", False),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=cfg["training"].get("pin_memory", False),
    )

    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    # ── Phase 6: Build model ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 6 — Building EfficientNet-B0 model")
    logger.info("=" * 60)
    from models.efficientnet_dr import build_model
    model = build_model(cfg)
    model = model.to(device)

    # ── Phase 7: Training ─────────────────────────────────────────────────────
    total_epochs = int(args.epochs or cfg["training"]["epochs"])
    default_stage1 = int(cfg["model"].get("freeze_backbone_epochs", 5))
    stage1_epochs = min(total_epochs, default_stage1)
    stage2_epochs = max(0, total_epochs - stage1_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_qwk": [], "val_f1": [],
    }

    best_qwk = -1.0
    best_epoch = 0
    patience = int(cfg["training"].get("early_stopping_patience", 7))
    patience_counter = 0
    best_model_path = PROJECT_ROOT / cfg["paths"]["best_model"]

    # ─── STAGE 1: Train head only ─────────────────────────────────────────────
    if stage1_epochs > 0:
        logger.info("=" * 60)
        logger.info("STAGE 1 -- Training classification head (%d epochs)", stage1_epochs)
        logger.info("=" * 60)
        model.freeze_backbone()
        optimizer = make_optimizer(model, cfg, stage=1)
        scheduler = make_scheduler(optimizer, cfg, stage1_epochs)

        for epoch in range(1, stage1_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = val_epoch(model, val_loader, criterion, device)
            qwk = compute_qwk(val_labels, val_preds)
            f1 = compute_macro_f1(val_labels, val_preds)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_qwk"].append(qwk)
            history["val_f1"].append(f1)

            logger.info(
                "Stage1 Epoch %d/%d | TrainLoss=%.4f TrainAcc=%.3f | "
                "ValLoss=%.4f ValAcc=%.3f QWK=%.4f F1=%.4f | %.1fs",
                epoch, stage1_epochs, train_loss, train_acc,
                val_loss, val_acc, qwk, f1, time.time() - t0
            )

            # Best model selection
            if qwk > best_qwk:
                best_qwk = qwk
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                logger.info("  [OK] New best model saved (QWK=%.4f)", best_qwk)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # ─── STAGE 2: Fine-tune backbone ─────────────────────────────────────────
    if stage2_epochs > 0:
        logger.info("=" * 60)
        logger.info("STAGE 2 -- Fine-tuning backbone (%d epochs)", stage2_epochs)
        logger.info("=" * 60)
        model.unfreeze_backbone(unfreeze_last_n_blocks=3)
        optimizer = make_optimizer(model, cfg, stage=2)
        scheduler = make_scheduler(optimizer, cfg, stage2_epochs)
        patience_counter = 0

        for epoch in range(1, stage2_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels = val_epoch(model, val_loader, criterion, device)
            qwk = compute_qwk(val_labels, val_preds)
            f1 = compute_macro_f1(val_labels, val_preds)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_qwk"].append(qwk)
            history["val_f1"].append(f1)

            logger.info(
                "Stage2 Epoch %d/%d | TrainLoss=%.4f TrainAcc=%.3f | "
                "ValLoss=%.4f ValAcc=%.3f QWK=%.4f F1=%.4f | %.1fs",
                epoch, stage2_epochs, train_loss, train_acc,
                val_loss, val_acc, qwk, f1, time.time() - t0
            )

            if qwk > best_qwk:
                best_qwk = qwk
                best_epoch = len(history["val_qwk"])
                torch.save(model.state_dict(), best_model_path)
                logger.info("  [OK] New best model saved (QWK=%.4f)", best_qwk)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping at stage2 epoch %d", epoch)
                break

    # ── Phase 9: Save model + metadata ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 9 — Saving model & metadata")
    logger.info("=" * 60)

    # Save training history
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save metadata
    metadata = {
        "model_type": "DRClassifier",
        "architecture": cfg["model"]["architecture"],
        "dataset": cfg["dataset"]["name"],
        "num_classes": 5,
        "class_names": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        "image_size": cfg["preprocessing"]["image_size"],
        "preprocessing_version": "1.0",
        "training_date": datetime.now().isoformat(),
        "best_epoch": best_epoch,
        "best_val_qwk": float(best_qwk),
        "total_epochs_trained": len(history["train_loss"]),
        "seed": seed,
        "device": str(device),
        "model_file": str(best_model_path),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save config snapshot alongside model
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Save class mapping
    class_map = {i: name for i, name in enumerate(["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"])}
    with open(model_dir / "class_mapping.json", "w") as f:
        json.dump(class_map, f, indent=2)

    logger.info("Training complete. Best QWK=%.4f at epoch %d", best_qwk, best_epoch)
    logger.info("Model saved to: %s", best_model_path)

    # Plot training curves
    _plot_training_curves(history, results_dir)

    # Phase 8: Run evaluation
    logger.info("=" * 60)
    logger.info("PHASE 8 — Running evaluation on validation set")
    logger.info("=" * 60)
    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    _run_evaluation(model, val_loader, device, results_dir, model_dir)

    return best_model_path


def _plot_training_curves(history: dict, results_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = list(range(1, len(history["train_loss"]) + 1))

        axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1b5ca8")
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#c47d00")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#1b5ca8")
        axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="#c47d00")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, history["val_qwk"], label="Val QWK", color="#2d7a2d")
        axes[2].plot(epochs, history["val_f1"], label="Val Macro F1", color="#b53000")
        axes[2].set_title("Validation Metrics")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        fig.suptitle("RetinaGuard — EfficientNet-B0 Training Curves", fontsize=12)
        plt.tight_layout()
        fig.savefig(results_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved to %s", results_dir / "training_curves.png")
    except Exception as e:
        logger.warning("Could not generate training curves: %s", e)


def _run_evaluation(model, val_loader, device, results_dir, model_dir):
    """Run full evaluation after training. See evaluate.py for standalone use."""
    try:
        import importlib.util, sys
        eval_path = Path(__file__).parent / "evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluate", eval_path)
        eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)
        eval_mod.evaluate_model(model, val_loader, device, results_dir, model_dir)
    except Exception as e:
        logger.warning("Evaluation step failed: %s", e)


# ==============================================================================
#  Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RetinaGuard DR classifier")
    parser.add_argument("--config", default=str(PYTHON_DIR / "config" / "config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--inspect_only", action="store_true",
                        help="Only run dataset inspection, no training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Config loaded: %s", args.config)

    if args.inspect_only:
        from dataset.aptos_loader import APTOSLoader
        loader = APTOSLoader(cfg)
        loader.inspect()
    else:
        train(cfg, args)
