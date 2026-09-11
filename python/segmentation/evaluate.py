"""
evaluate.py
===========
Rigorous, Unbiased Evaluation of Dual-Head U-Net on the Unseen IDRiD Test Set (N=27).

Strict Scientific Protocol:
- Evaluates on the official 27 test images never seen during training or tuning.
- Decouples Lesion Pathology (MA, HE, EX, SE) from Normal Anatomy (Optic Disc).
- Computes Dice, IoU, Precision, Recall, Specificity.
- Reports positive image counts, empty mask counts, and qualitative comparisons.
- Exports structured metrics to metrics.json and qualitative comparison plot to reports/.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from python.segmentation.dataset import IDRiDSegmentationDataset
from python.segmentation.unet import DualHeadUNet

EXP_DIR = ROOT / "models" / "experiments" / "lesion_segmentation"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LESION_NAMES = ["Microaneurysms (MA)", "Haemorrhages (HE)", "Hard Exudates (EX)", "Soft Exudates (SE)"]
ANATOMY_NAMES = ["Optic Disc (OD)"]


def evaluate(threshold: float = 0.5):
    print("=" * 70)
    print("  RETINAGUARD — UNBIASED TEST EVALUATION (IDRiD TEST SET N=27)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idrid_root = ROOT / "data" / "idrid" / "A. Segmentation"
    test_ds = IDRiDSegmentationDataset(idrid_root, split="test", image_size=256, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"[*] Total test images: {len(test_ds)} (Official ISBI 2018 Test Set IDs 55 to 81)")

    # Load trained model
    ckpt_path = EXP_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {ckpt_path}")

    model = DualHeadUNet(
        in_channels=3,
        num_lesion_classes=4,
        num_anatomy_classes=1,
        base_features=32,
        export_mode=False
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[*] Loaded best checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    # Metrics accumulators
    class_stats = {
        "MA": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pos_images": 0, "empty_images": 0, "name": "Microaneurysms"},
        "HE": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pos_images": 0, "empty_images": 0, "name": "Haemorrhages"},
        "EX": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pos_images": 0, "empty_images": 0, "name": "Hard Exudates"},
        "SE": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pos_images": 0, "empty_images": 0, "name": "Soft Exudates"},
        "OD": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "pos_images": 0, "empty_images": 0, "name": "Optic Disc (Anatomy)"},
    }

    per_image_dices = {k: [] for k in class_stats.keys()}
    qualitative_samples = []

    with torch.no_grad():
        for idx, (img_tensor, lesion_target, anatomy_target, meta) in enumerate(test_loader):
            img_tensor = img_tensor.to(device)
            l_logits, a_logits = model(img_tensor)

            l_probs = torch.sigmoid(l_logits).cpu().numpy()[0]   # [4, 256, 256]
            a_probs = torch.sigmoid(a_logits).cpu().numpy()[0]   # [1, 256, 256]

            l_pred = (l_probs > threshold).astype(np.uint8)
            a_pred = (a_probs > threshold).astype(np.uint8)

            l_gt = lesion_target.numpy()[0].astype(np.uint8)      # [4, 256, 256]
            a_gt = anatomy_target.numpy()[0].astype(np.uint8)     # [1, 256, 256]

            # Collect qualitative samples for images 0, 5, 12, 18
            if idx in [0, 5, 12, 18]:
                # Denormalize image for visualization
                raw_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                raw_img = np.clip((raw_img * std + mean) * 255, 0, 255).astype(np.uint8)

                qualitative_samples.append({
                    "image_id": meta["image_id"][0],
                    "raw_img": raw_img,
                    "gt_lesions": l_gt,
                    "pred_lesions": l_pred,
                    "gt_od": a_gt[0],
                    "pred_od": a_pred[0]
                })

            # Evaluate 4 lesion classes
            keys = ["MA", "HE", "EX", "SE"]
            for c_idx, key in enumerate(keys):
                p = l_pred[c_idx]
                g = l_gt[c_idx]

                pos = int(np.sum(g > 0))
                if pos > 0:
                    class_stats[key]["pos_images"] += 1
                else:
                    class_stats[key]["empty_images"] += 1

                tp = int(np.sum((p == 1) & (g == 1)))
                fp = int(np.sum((p == 1) & (g == 0)))
                fn = int(np.sum((p == 0) & (g == 1)))
                tn = int(np.sum((p == 0) & (g == 0)))

                class_stats[key]["tp"] += tp
                class_stats[key]["fp"] += fp
                class_stats[key]["fn"] += fn
                class_stats[key]["tn"] += tn

                img_dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
                per_image_dices[key].append(float(img_dice))

            # Evaluate Optic Disc (OD)
            p_od = a_pred[0]
            g_od = a_gt[0]
            if np.sum(g_od > 0) > 0:
                class_stats["OD"]["pos_images"] += 1
            else:
                class_stats["OD"]["empty_images"] += 1

            tp_od = int(np.sum((p_od == 1) & (g_od == 1)))
            fp_od = int(np.sum((p_od == 1) & (g_od == 0)))
            fn_od = int(np.sum((p_od == 0) & (g_od == 1)))
            tn_od = int(np.sum((p_od == 0) & (g_od == 0)))

            class_stats["OD"]["tp"] += tp_od
            class_stats["OD"]["fp"] += fp_od
            class_stats["OD"]["fn"] += fn_od
            class_stats["OD"]["tn"] += tn_od
            img_dice_od = (2.0 * tp_od + 1e-6) / (2.0 * tp_od + fp_od + fn_od + 1e-6)
            per_image_dices["OD"].append(float(img_dice_od))

    # Compute final metrics
    results = {}
    print("\n" + "=" * 80)
    print(f"{'Target Class':<24} {'Category':<10} {'Pos/Empty':<12} {'Dice':<8} {'IoU':<8} {'Precision':<10} {'Recall'}")
    print("=" * 80)

    for key, stat in class_stats.items():
        tp = stat["tp"]
        fp = stat["fp"]
        fn = stat["fn"]
        tn = stat["tn"]

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        category = "Anatomy" if key == "OD" else "Pathology"
        pos_empty = f"{stat['pos_images']}/{stat['empty_images']}"

        print(f"{stat['name']:<24} {category:<10} {pos_empty:<12} {dice:<8.4f} {iou:<8.4f} {precision:<10.4f} {recall:<.4f}")

        results[key] = {
            "name": stat["name"],
            "category": category,
            "positive_test_images": stat["pos_images"],
            "empty_mask_test_images": stat["empty_images"],
            "dice": round(float(dice), 4),
            "iou": round(float(iou), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(float(specificity), 4),
            "mean_per_image_dice": round(float(np.mean(per_image_dices[key])), 4)
        }

    # Macro averages
    lesion_keys = ["MA", "HE", "EX", "SE"]
    results["summary"] = {
        "dataset": "IDRiD ISBI 2018 (Part A)",
        "num_test_images": len(test_ds),
        "mean_lesion_dice": round(float(np.mean([results[k]["dice"] for k in lesion_keys])), 4),
        "mean_lesion_iou": round(float(np.mean([results[k]["iou"] for k in lesion_keys])), 4),
        "optic_disc_dice": results["OD"]["dice"],
        "optic_disc_iou": results["OD"]["iou"],
        "disclaimer": "RetinaGuard AI lesion segmentation is a research decision-support prototype evaluated on 27 test images. It does not provide an autonomous clinical diagnosis."
    }

    print("=" * 80)
    print(f"[*] Macro Lesion Dice : {results['summary']['mean_lesion_dice']:.4f}")
    print(f"[*] Optic Disc Dice   : {results['summary']['optic_disc_dice']:.4f}")

    # Save metrics JSON
    metrics_file = EXP_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Test metrics saved to: {metrics_file}")

    # Render qualitative visualization
    print("\n[*] Generating qualitative comparison figure...")
    render_qualitative_figure(qualitative_samples)

    return results


def render_qualitative_figure(samples: List[Dict]):
    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, 0)

    for i, s in enumerate(samples):
        # 1. Original Image
        axes[i, 0].imshow(s["raw_img"])
        axes[i, 0].set_title(f"Test Case: {s['image_id']}\nOriginal Fundus")
        axes[i, 0].axis("off")

        # 2. Ground Truth Lesions composite
        # Color encoding: MA (red), HE (orange), EX (yellow), SE (cyan)
        h, w, _ = s["raw_img"].shape
        gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gt_vis[s["gt_lesions"][0] > 0] = [255, 0, 0]       # MA
        gt_vis[s["gt_lesions"][1] > 0] = [255, 140, 0]     # HE
        gt_vis[s["gt_lesions"][2] > 0] = [255, 255, 0]     # EX
        gt_vis[s["gt_lesions"][3] > 0] = [0, 255, 255]     # SE

        # Blend with fundus
        gt_blend = cv2.addWeighted(s["raw_img"], 0.6, gt_vis, 0.4, 0) if np.sum(gt_vis) > 0 else s["raw_img"]
        axes[i, 1].imshow(gt_blend)
        axes[i, 1].set_title("Expert Ground Truth\n(MA=Red, HE=Org, EX=Yel, SE=Cyan)")
        axes[i, 1].axis("off")

        # 3. Predicted Lesions composite
        pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
        pred_vis[s["pred_lesions"][0] > 0] = [255, 0, 0]
        pred_vis[s["pred_lesions"][1] > 0] = [255, 140, 0]
        pred_vis[s["pred_lesions"][2] > 0] = [255, 255, 0]
        pred_vis[s["pred_lesions"][3] > 0] = [0, 255, 255]

        pred_blend = cv2.addWeighted(s["raw_img"], 0.6, pred_vis, 0.4, 0) if np.sum(pred_vis) > 0 else s["raw_img"]
        axes[i, 2].imshow(pred_blend)
        axes[i, 2].set_title("AI Predicted Lesions\n(Dual-Head U-Net)")
        axes[i, 2].axis("off")

        # 4. Optic Disc (GT in Green, Pred in Magenta)
        od_vis = np.zeros((h, w, 3), dtype=np.uint8)
        od_vis[s["gt_od"] > 0] = [0, 255, 0]        # Green for GT
        od_vis[s["pred_od"] > 0] = [255, 0, 255]    # Magenta for Pred
        od_blend = cv2.addWeighted(s["raw_img"], 0.6, od_vis, 0.4, 0) if np.sum(od_vis) > 0 else s["raw_img"]
        axes[i, 3].imshow(od_blend)
        axes[i, 3].set_title("Optic Disc Localization\n(GT=Green, Pred=Magenta)")
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_path = REPORTS_DIR / "lesion_segmentation_predictions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Qualitative plot saved to: {out_path}")


if __name__ == "__main__":
    evaluate()
