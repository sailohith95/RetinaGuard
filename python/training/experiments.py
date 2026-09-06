"""
RetinaGuard — Systematic Experiment Runner & Tracker
=====================================================
Executes and logs comparative experiments to improve model performance
on minority classes (especially Grade 4 Proliferative DR) and overall QWK.

Tracks for every experiment:
  - experiment ID
  - model architecture
  - loss function
  - augmentation configuration
  - learning rates (stage 1 & 2)
  - batch size
  - epochs (stage 1 & 2)
  - class balancing strategy
  - validation accuracy
  - macro F1
  - QWK
  - per-class F1 (especially Grade 4)
  - referable DR sensitivity & specificity
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).parent
PYTHON_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.aptos_loader import APTOSLoader
from dataset.aptos_dataset import APTOSDataset
from preprocessing.retinal_preprocessor import RetinalPreprocessor
from models.efficientnet_dr import build_model, DRClassifier
from training.train import set_seed, get_device, train_epoch, val_epoch, compute_qwk, compute_macro_f1
from training.evaluate import evaluate_model
from training.losses import build_loss_function

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinaguard.experiments")

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


class ExperimentTracker:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.experiments: List[Dict] = []
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.experiments = json.load(f)
            except Exception:
                self.experiments = []

    def record(self, exp_data: Dict):
        # Check if exp_id exists, replace or append
        existing = next((i for i, e in enumerate(self.experiments) if e["exp_id"] == exp_data["exp_id"]), None)
        if existing is not None:
            self.experiments[existing] = exp_data
        else:
            self.experiments.append(exp_data)
        
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)
        self._write_markdown_table()

    def _write_markdown_table(self):
        md_path = self.log_path.parent / "experiments_table.md"
        headers = [
            "Exp ID", "Loss", "Sampler", "St1/St2 Ep", "Acc (%)", "QWK",
            "Macro F1", "G0 F1", "G1 F1", "G2 F1", "G3 F1", "G4 F1", "Ref Sens (%)", "Ref Spec (%)"
        ]
        rows = []
        for e in self.experiments:
            m = e["metrics"]
            pc = m["per_class"]
            row = [
                e["exp_id"],
                e["loss"],
                e["sampler"],
                f"{e['stage1_epochs']}+{e['stage2_epochs']}",
                f"{m['accuracy']*100:.1f}",
                f"{m['qwk']:.4f}",
                f"{m['macro_f1']:.4f}",
                f"{pc['No DR']['f1']:.3f}",
                f"{pc['Mild DR']['f1']:.3f}",
                f"{pc['Moderate DR']['f1']:.3f}",
                f"{pc['Severe DR']['f1']:.3f}",
                f"{pc['Proliferative DR']['f1']:.3f}",
                f"{m['referable_sensitivity']*100:.1f}",
                f"{m['referable_specificity']*100:.1f}"
            ]
            rows.append(" | ".join(row))

        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for r in rows:
            table += "| " + r + " |\n"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# RetinaGuard — Experiment Results Log\n\n" + table)
        logger.info("Updated experiments table at %s", md_path)


def run_experiment(
    exp_id: str,
    loss_type: str,
    use_weighted_sampler: bool = False,
    stage1_epochs: int = 3,
    stage2_epochs: int = 0,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    gamma: float = 2.0,
    ordinal_lambda: float = 0.25,
    custom_weights_ratio: Optional[dict] = None,
    vertical_flip: bool = True,
    rotation_degrees: int = 15,
    batch_size: int = 32,
    sampler_power: float = 0.5,
    grade4_boost: float = 1.0,
    seed: int = 42
) -> Dict:
    set_seed(seed)
    device = get_device()
    logger.info("=" * 70)
    logger.info("RUNNING EXPERIMENT: %s", exp_id)
    logger.info("Loss: %s | Sampler: %s | St1: %d ep | St2: %d ep",
                loss_type, "WeightedRandomSampler" if use_weighted_sampler else "Standard",
                stage1_epochs, stage2_epochs)
    logger.info("=" * 70)

    # 1. Config setup
    with open(PYTHON_DIR / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)
    cfg["augmentation"]["vertical_flip"] = vertical_flip
    cfg["augmentation"]["rotation_degrees"] = rotation_degrees

    # 2. Dataset & Preprocessor
    prep = RetinalPreprocessor(cfg)
    loader = APTOSLoader(cfg)
    loader.load()
    train_ds, val_ds = loader.get_splits(prep)

    train_labels = train_ds.get_labels()
    train_counts = np.bincount(train_labels, minlength=5)
    class_weights = loader.get_class_weights(train_labels)

    # If custom weights specified (e.g. boost Grade 4)
    if custom_weights_ratio is not None:
        for c, mult in custom_weights_ratio.items():
            class_weights[c] *= mult
        # Re-normalize
        class_weights = class_weights / class_weights.sum() * 5.0

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 3. Build Loss Function
    criterion = build_loss_function(
        loss_name=loss_type,
        class_weights=weight_tensor,
        train_counts=train_counts.tolist(),
        gamma=gamma,
        ordinal_lambda=ordinal_lambda
    )

    # 4. Sampler & DataLoaders
    if use_weighted_sampler:
        # Parameterized minority-class balancing
        class_sample_counts = np.bincount(train_labels, minlength=5)
        # Power-scaled inverse frequency: power=0.5 for moderate/sqrt, power=1.0 for full inverse frequency
        sample_weights_per_class = (1.0 / np.maximum(class_sample_counts, 1).astype(float)) ** sampler_power
        if grade4_boost != 1.0:
            sample_weights_per_class[4] *= grade4_boost
            logger.info("Applied configurable Grade 4 boost multiplier: %.2fx", grade4_boost)
        logger.info("Sampler per-class weights (power=%.2f, g4_boost=%.2f): %s",
                    sampler_power, grade4_boost, [f"{w:.5f}" for w in sample_weights_per_class])
        sample_weights = sample_weights_per_class[train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 5. Model
    model = build_model(cfg).to(device)

    # Save directory for this experiment
    exp_dir = PROJECT_ROOT / "models" / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_ckpt = exp_dir / "best_model.pt"

    best_qwk = -1.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_qwk": [], "val_f1": []}

    # ─── Stage 1: Head training ─────────────────────────────────────────────
    if stage1_epochs > 0:
        logger.info("STAGE 1: Training head only (%d epochs, lr=%.1e)...", stage1_epochs, lr_stage1)
        model.freeze_backbone()
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_stage1, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)

        for ep in range(1, stage1_epochs + 1):
            t0 = time.time()
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_preds, v_labels = val_epoch(model, val_loader, criterion, device)
            qwk = compute_qwk(v_labels, v_preds)
            f1 = compute_macro_f1(v_labels, v_preds)
            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_qwk"].append(qwk)
            history["val_f1"].append(f1)

            logger.info("  St1 Ep %d/%d | TrainLoss=%.4f TrainAcc=%.3f | ValLoss=%.4f ValAcc=%.3f QWK=%.4f F1=%.4f | %.1fs",
                        ep, stage1_epochs, t_loss, t_acc, v_loss, v_acc, qwk, f1, time.time() - t0)

            if qwk > best_qwk:
                best_qwk = qwk
                best_epoch = ep
                torch.save(model.state_dict(), exp_ckpt)
                logger.info("    [OK] New best checkpoint saved (QWK=%.4f)", best_qwk)

    # ─── Stage 2: Fine-tuning top backbone blocks ───────────────────────────
    if stage2_epochs > 0:
        logger.info("STAGE 2: Fine-tuning backbone (%d epochs, lr=%.1e)...", stage2_epochs, lr_stage2)
        model.unfreeze_backbone(unfreeze_last_n_blocks=3)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr_stage2, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)

        for ep in range(1, stage2_epochs + 1):
            t0 = time.time()
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_preds, v_labels = val_epoch(model, val_loader, criterion, device)
            qwk = compute_qwk(v_labels, v_preds)
            f1 = compute_macro_f1(v_labels, v_preds)
            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_qwk"].append(qwk)
            history["val_f1"].append(f1)

            logger.info("  St2 Ep %d/%d | TrainLoss=%.4f TrainAcc=%.3f | ValLoss=%.4f ValAcc=%.3f QWK=%.4f F1=%.4f | %.1fs",
                        ep, stage2_epochs, t_loss, t_acc, v_loss, v_acc, qwk, f1, time.time() - t0)

            if qwk > best_qwk:
                best_qwk = qwk
                best_epoch = stage1_epochs + ep
                torch.save(model.state_dict(), exp_ckpt)
                logger.info("    [OK] New best checkpoint saved (QWK=%.4f)", best_qwk)

    # 6. Evaluate Best Checkpoint
    logger.info("Evaluating best model checkpoint (epoch %d, QWK=%.4f)...", best_epoch, best_qwk)
    model.load_state_dict(torch.load(exp_ckpt, map_location=device, weights_only=True))
    metrics = evaluate_model(model, val_loader, device, exp_dir, exp_dir)

    exp_result = {
        "exp_id": exp_id,
        "model": "EfficientNet-B0",
        "loss": loss_type,
        "sampler": "WeightedRandomSampler" if use_weighted_sampler else "Standard",
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "lr_stage1": lr_stage1,
        "lr_stage2": lr_stage2,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "checkpoint_path": str(exp_ckpt),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }

    tracker = ExperimentTracker(PROJECT_ROOT / "results" / "experiments_log.json")
    tracker.record(exp_result)

    logger.info("Experiment %s complete. Accuracy: %.2f%% | QWK: %.4f | Macro F1: %.4f | Grade 4 F1: %.4f",
                exp_id, metrics["accuracy"] * 100, metrics["qwk"], metrics["macro_f1"],
                metrics["per_class"]["Proliferative DR"]["f1"])
    return exp_result
