"""
RetinaGuard — Quick Dataset + DataLoader Smoke Test
=====================================================
Verifies the full data pipeline (loader → split → preprocessor → dataloader)
without training. Much faster than running train.py.

Usage:
    python python/test_dataloader.py
"""

import sys, logging
from pathlib import Path
import yaml, numpy as np

PYTHON_DIR = Path(__file__).parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinaguard.test_dl")

def main():
    cfg_path = PYTHON_DIR / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    print("\n" + "=" * 60)
    print("  RetinaGuard — DataLoader Smoke Test")
    print("=" * 60)

    # Load dataset (from cache)
    from dataset.aptos_loader import APTOSLoader
    loader = APTOSLoader(cfg)
    loader.load()
    print(f"\nDataset loaded. Columns: image='{loader._image_col}', label='{loader._label_col}'")

    # Build preprocessor
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    prep = RetinalPreprocessor(cfg)
    print("Preprocessor built.")

    # Get splits
    train_ds, val_ds = loader.get_splits(prep)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Class weights
    weights = loader.get_class_weights(train_ds.get_labels())
    print(f"Class weights: {[round(float(w),3) for w in weights]}")

    # Test one training batch
    import torch
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0
    )
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"\nTrain batch: imgs={tuple(batch_imgs.shape)}, "
          f"labels={batch_labels.tolist()}, "
          f"img dtype={batch_imgs.dtype}")

    # Test one val batch
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0
    )
    vimgs, vlabels = next(iter(val_loader))
    print(f"Val   batch: imgs={tuple(vimgs.shape)}, labels={vlabels.tolist()}")

    print("\nData pipeline smoke test PASSED.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
