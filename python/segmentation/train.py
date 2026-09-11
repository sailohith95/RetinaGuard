"""
train.py
========
Training script for Dual-Head U-Net on IDRiD Retinal Fundus Dataset.

Strict Constraints:
- Completely separate experiment folder: models/experiments/lesion_segmentation/
- Never touches or overwrites models/aptos_efficientnet/ or EXP-001
- Official 27 test images are strictly held out for final evaluation
- Multi-task BCE + Soft Dice loss with sparsity-weighted lesion terms
- Exports high-speed ONNX model (lesion_unet.onnx) for inference & MATLAB
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from python.segmentation.dataset import IDRiDSegmentationDataset
from python.segmentation.unet import DualHeadUNet

EXP_DIR = ROOT / "models" / "experiments" / "lesion_segmentation"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num_channels = probs.shape[1]
        dice_losses = []
        for c in range(num_channels):
            p = probs[:, c, :, :].contiguous().view(-1)
            t = targets[:, c, :, :].contiguous().view(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_losses.append(1.0 - dice)
        return torch.stack(dice_losses)


class MultiTargetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = SoftDiceLoss(smooth=1.0)
        # Weights for MA, HE, EX, SE (MA has sparse pixel count, given higher weight)
        self.lesion_weights = torch.tensor([1.5, 1.0, 1.0, 1.0])

    def forward(
        self,
        lesion_logits: torch.Tensor,
        anatomy_logits: torch.Tensor,
        lesion_targets: torch.Tensor,
        anatomy_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = lesion_logits.device
        weights = self.lesion_weights.to(device)

        # 1. Lesion Loss
        bce_lesion = self.bce(lesion_logits, lesion_targets).mean(dim=[0, 2, 3])  # [4]
        dice_lesion = self.dice(lesion_logits, lesion_targets)                    # [4]
        combined_lesion = (0.5 * bce_lesion + 0.5 * dice_lesion) * weights
        loss_lesion = combined_lesion.sum()

        # 2. Anatomy Loss (Optic Disc)
        bce_anatomy = self.bce(anatomy_logits, anatomy_targets).mean()
        dice_anatomy = self.dice(anatomy_logits, anatomy_targets).mean()
        loss_anatomy = 0.5 * bce_anatomy + 0.5 * dice_anatomy

        # Total multi-task loss
        total_loss = loss_lesion + 0.5 * loss_anatomy

        details = {
            "loss_total": float(total_loss.item()),
            "loss_lesion": float(loss_lesion.item()),
            "loss_anatomy": float(loss_anatomy.item()),
            "dice_ma": float(1.0 - dice_lesion[0].item()),
            "dice_he": float(1.0 - dice_lesion[1].item()),
            "dice_ex": float(1.0 - dice_lesion[2].item()),
            "dice_se": float(1.0 - dice_lesion[3].item()),
            "dice_od": float(1.0 - dice_anatomy.item()),
        }
        return total_loss, details


def compute_eval_dice(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> List[float]:
    preds = (probs > threshold).float()
    num_channels = preds.shape[1]
    dices = []
    for c in range(num_channels):
        p = preds[:, c, :, :].contiguous().view(-1)
        t = targets[:, c, :, :].contiguous().view(-1)
        intersection = (p * t).sum()
        dice = (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)
        dices.append(float(dice.item()))
    return dices


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Dual-Head U-Net on device: {device}")

    idrid_root = ROOT / "data" / "idrid" / "A. Segmentation"
    full_train_ds = IDRiDSegmentationDataset(idrid_root, split="train", image_size=256, augment=True)
    
    # Stratified/random 80/20 train/val split of the 54 training images
    n_total = len(full_train_ds)
    indices = list(range(n_total))
    random.shuffle(indices)
    n_val = 10
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_ds = Subset(full_train_ds, train_indices)
    # Val dataset without augmentation
    val_base_ds = IDRiDSegmentationDataset(idrid_root, split="train", image_size=256, augment=False)
    val_ds = Subset(val_base_ds, val_indices)

    print(f"[*] Train images: {len(train_ds)} | Validation images: {len(val_ds)}")
    print(f"[*] Note: Official 27 test images held untouched in data/idrid/... for final evaluation.")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = DualHeadUNet(
        in_channels=3,
        num_lesion_classes=4,
        num_anatomy_classes=1,
        base_features=32,
        export_mode=False
    ).to(device)

    criterion = MultiTargetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_val_dice = 0.0
    best_epoch = -1
    log_rows = []

    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<10} {'Mean Lesion Dice':<18} {'OD Dice':<10} {'LR'}")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for imgs, lesions, anatomies, _ in train_loader:
            imgs = imgs.to(device)
            lesions = lesions.to(device)
            anatomies = anatomies.to(device)

            optimizer.zero_grad()
            l_logits, a_logits = model(imgs)
            loss, _ = criterion(l_logits, a_logits, lesions, anatomies)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        mean_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        val_lesion_dices = []
        val_od_dices = []

        with torch.no_grad():
            for imgs, lesions, anatomies, _ in val_loader:
                imgs = imgs.to(device)
                lesions = lesions.to(device)
                anatomies = anatomies.to(device)

                l_logits, a_logits = model(imgs)
                loss, _ = criterion(l_logits, a_logits, lesions, anatomies)
                val_losses.append(loss.item())

                l_probs = torch.sigmoid(l_logits)
                a_probs = torch.sigmoid(a_logits)

                l_dice = compute_eval_dice(l_probs, lesions)
                a_dice = compute_eval_dice(a_probs, anatomies)

                val_lesion_dices.append(l_dice)
                val_od_dices.append(a_dice[0])

        mean_val_loss = float(np.mean(val_losses))
        avg_lesion_dices = np.mean(val_lesion_dices, axis=0)  # [4]
        mean_lesion_dice = float(np.mean(avg_lesion_dices))
        mean_od_dice = float(np.mean(val_od_dices))
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:<8} {mean_train_loss:<12.4f} {mean_val_loss:<10.4f} "
              f"{mean_lesion_dice:<18.4f} {mean_od_dice:<10.4f} {current_lr:.6f}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "mean_lesion_dice": mean_lesion_dice,
            "od_dice": mean_od_dice,
            "dice_ma": float(avg_lesion_dices[0]),
            "dice_he": float(avg_lesion_dices[1]),
            "dice_ex": float(avg_lesion_dices[2]),
            "dice_se": float(avg_lesion_dices[3]),
            "lr": current_lr
        })

        # Save best checkpoint on combined validation Dice
        overall_score = 0.7 * mean_lesion_dice + 0.3 * mean_od_dice
        if overall_score > best_val_dice:
            best_val_dice = overall_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "mean_lesion_dice": mean_lesion_dice,
                "mean_od_dice": mean_od_dice,
            }, EXP_DIR / "best_model.pt")

    print("=" * 70)
    print(f"[OK] Training complete. Best epoch: {best_epoch} with validation score: {best_val_dice:.4f}")

    # Save final model checkpoint
    torch.save(model.state_dict(), EXP_DIR / "final_model.pt")

    # Save training log
    with open(EXP_DIR / "training_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)

    # Save config
    config = {
        "architecture": "DualHeadUNet",
        "dataset": "IDRiD ISBI 2018 Challenge (Part A)",
        "image_size": 256,
        "input_channels": 3,
        "lesion_classes": ["MA", "HE", "EX", "SE"],
        "anatomy_classes": ["OD"],
        "num_epochs": num_epochs,
        "batch_size": 4,
        "base_lr": 1e-3,
        "optimizer": "AdamW",
        "loss": "BCE + SoftDiceLoss",
        "best_epoch": best_epoch,
        "best_val_score": best_val_dice
    }
    with open(EXP_DIR / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Export to ONNX
    print("\n[*] Exporting best model to ONNX format...")
    export_model = DualHeadUNet(
        in_channels=3,
        num_lesion_classes=4,
        num_anatomy_classes=1,
        base_features=32,
        export_mode=True
    )
    checkpoint = torch.load(EXP_DIR / "best_model.pt", map_location="cpu")
    export_model.load_state_dict(checkpoint["model_state_dict"])
    export_model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    onnx_path = EXP_DIR / "lesion_unet.onnx"

    torch.onnx.export(
        export_model,
        dummy_input,
        str(onnx_path),
        input_names=["input_image"],
        output_names=["sigmoid_masks"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "sigmoid_masks": {0: "batch_size"}
        },
        opset_version=13
    )

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[OK] ONNX export saved to: {onnx_path} ({onnx_size_mb:.2f} MB)")
    print(f"[OK] All training artifacts stored in: {EXP_DIR}")


if __name__ == "__main__":
    train()
