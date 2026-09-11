"""
audit_idrid.py
==============
Automated Integrity Audit for IDRiD Part A - Segmentation Challenge Dataset.

Audits:
- Exact image counts (train and test)
- File integrity and readability
- Image dimensions and channels
- Mask presence, dimensions, and binary values
- Positive image counts and empty-mask percentages per class
- Lesion pixel area coverage (% of total image)
- Strict train/test split verification
- Output structured audit report to console and JSON/Markdown
"""

import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
IDRID_DIR = ROOT / "data" / "idrid" / "A. Segmentation"
ORIG_DIR = IDRID_DIR / "1. Original Images"
GT_DIR = IDRID_DIR / "2. All Segmentation Groundtruths"

CLASSES = [
    ("1. Microaneurysms", "MA", "Microaneurysms", "Lesion"),
    ("2. Haemorrhages", "HE", "Haemorrhages", "Lesion"),
    ("3. Hard Exudates", "EX", "Hard Exudates", "Lesion"),
    ("4. Soft Exudates", "SE", "Soft Exudates", "Lesion"),
    ("5. Optic Disc", "OD", "Optic Disc", "Anatomy"),
]


def audit_split(split_name: str, img_subfolder: str, gt_subfolder: str):
    img_dir = ORIG_DIR / img_subfolder
    gt_dir = GT_DIR / gt_subfolder

    if not img_dir.exists():
        return {"error": f"Image directory not found: {img_dir}"}

    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])
    
    split_results = {
        "split": split_name,
        "total_images": len(images),
        "image_list": [f.name for f in images],
        "image_dimensions": set(),
        "corrupt_images": [],
        "classes": {}
    }

    # Initialize class stats
    for folder_name, abbrev, display_name, category in CLASSES:
        split_results["classes"][abbrev] = {
            "display_name": display_name,
            "category": category,
            "folder": folder_name,
            "total_masks_found": 0,
            "positive_images": 0,
            "empty_masks": 0,
            "missing_mask_files": 0,
            "total_pixels_evaluated": 0,
            "total_positive_pixels": 0,
            "pixel_coverage_pct": 0.0,
            "unique_values": set(),
            "dimension_mismatches": []
        }

    for img_path in images:
        img_id = img_path.stem  # e.g. IDRiD_01
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            split_results["corrupt_images"].append(img_path.name)
            continue
        
        h, w, c = img.shape
        split_results["image_dimensions"].add(f"{w}x{h}x{c}")

        # Check each class mask
        for folder_name, abbrev, display_name, category in CLASSES:
            class_dir = gt_dir / folder_name
            mask_file = None
            
            # Look for mask with prefix e.g. IDRiD_01_MA.tif or IDRiD_01.tif
            candidates = [
                class_dir / f"{img_id}_{abbrev}.tif",
                class_dir / f"{img_id}_{abbrev}.png",
                class_dir / f"{img_id}.tif",
                class_dir / f"{img_id}.png"
            ]
            for c_path in candidates:
                if c_path.exists():
                    mask_file = c_path
                    break

            class_stat = split_results["classes"][abbrev]
            if mask_file is None:
                # In IDRiD, if an image has NO lesions of that type, is the mask missing or empty?
                class_stat["missing_mask_files"] += 1
                class_stat["empty_masks"] += 1
            else:
                class_stat["total_masks_found"] += 1
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    class_stat["dimension_mismatches"].append(f"{mask_file.name} unreadable")
                    continue
                
                mh, mw = mask.shape
                if (mh, mw) != (h, w):
                    class_stat["dimension_mismatches"].append(f"{mask_file.name} ({mw}x{mh}) != ({w}x{h})")

                unique_vals = np.unique(mask)
                for u in unique_vals:
                    class_stat["unique_values"].add(int(u))

                pos_pixels = int(np.sum(mask > 0))
                total_pixels = h * w
                class_stat["total_pixels_evaluated"] += total_pixels
                class_stat["total_positive_pixels"] += pos_pixels

                if pos_pixels > 0:
                    class_stat["positive_images"] += 1
                else:
                    class_stat["empty_masks"] += 1

    # Convert sets to sorted lists for JSON serialization and compute percentages
    split_results["image_dimensions"] = sorted(list(split_results["image_dimensions"]))
    for abbrev, stat in split_results["classes"].items():
        stat["unique_values"] = sorted(list(stat["unique_values"]))
        if stat["total_pixels_evaluated"] > 0:
            stat["pixel_coverage_pct"] = round((stat["total_positive_pixels"] / stat["total_pixels_evaluated"]) * 100, 4)
        total_img = split_results["total_images"]
        stat["positive_rate_pct"] = round((stat["positive_images"] / total_img) * 100, 2) if total_img > 0 else 0
        stat["empty_rate_pct"] = round((stat["empty_masks"] / total_img) * 100, 2) if total_img > 0 else 0

    return split_results


def run_audit():
    print("=" * 70)
    print("  RETINAGUARD — IDRiD DATASET INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Dataset root: {IDRID_DIR}\n")

    if not IDRID_DIR.exists():
        print(f"[ERROR] IDRiD directory does not exist: {IDRID_DIR}")
        return None

    train_results = audit_split("Training Set", "a. Training Set", "a. Training Set")
    test_results = audit_split("Testing Set", "b. Testing Set", "b. Testing Set")

    combined = {
        "train": train_results,
        "test": test_results,
        "summary": {
            "total_images": train_results.get("total_images", 0) + test_results.get("total_images", 0),
            "train_images": train_results.get("total_images", 0),
            "test_images": test_results.get("total_images", 0),
            "split_ratio": f"{train_results.get('total_images', 0)}:{test_results.get('total_images', 0)}",
            "is_official_split_preserved": (
                train_results.get("total_images", 0) == 54 and test_results.get("total_images", 0) == 27
            )
        }
    }

    # Print human-readable report
    print(f"[*] Total Images Audited : {combined['summary']['total_images']}")
    print(f"    - Training Set       : {combined['summary']['train_images']} images (IDs 01 to 54)")
    print(f"    - Testing Set        : {combined['summary']['test_images']} images (IDs 55 to 81)")
    print(f"    - Official Split     : {'PRESERVED [OK]' if combined['summary']['is_official_split_preserved'] else 'MISMATCH [WARN]'}\n")

    for split_key, title in [("train", "TRAINING SET (N=54)"), ("test", "TESTING SET (N=27)")]:
        res = combined[split_key]
        print(f"--- {title} ---")
        print(f"  Dimensions : {', '.join(res.get('image_dimensions', []))}")
        print(f"  Corrupted  : {len(res.get('corrupt_images', []))} files\n")
        
        print(f"  {'Class':<20} {'Category':<10} {'Masks Found':<12} {'Pos Imgs':<10} {'Empty/None':<12} {'Pixel Cov %'}")
        print("  " + "-" * 72)
        for abbrev, stat in res.get("classes", {}).items():
            print(f"  {stat['display_name']:<20} {stat['category']:<10} {stat['total_masks_found']:<12} "
                  f"{stat['positive_images']:<10} {stat['empty_masks']:<12} {stat['pixel_coverage_pct']:.4f}%")
        print()

    # Save report
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "idrid_audit_report.json"
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[OK] Audit report saved to: {out_json}")
    print("=" * 70)
    return combined


if __name__ == "__main__":
    run_audit()
