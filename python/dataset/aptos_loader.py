"""
RetinaGuard — APTOS Dataset Loader
===================================
Loads the bumbledeep/aptos dataset from HuggingFace with local caching.
Handles dataset inspection, class-stratified splitting, and dataloader creation.

Usage:
    from dataset.aptos_loader import APTOSLoader
    loader = APTOSLoader(cfg)
    train_ds, val_ds = loader.get_splits()
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class APTOSLoader:
    """
    Manages loading, caching, splitting, and inspection of the APTOS dataset.

    The dataset is downloaded once and cached locally. Subsequent runs read
    from cache without any network access, which is critical for offline demo.
    """

    # ICDR severity labels — must NOT be modified
    CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    NUM_CLASSES = 5

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ds_cfg = cfg["dataset"]
        self.root = Path(cfg.get("_root", "."))

        self.cache_dir = self.root / self.ds_cfg["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.split_file = self.root / cfg["paths"]["split_file"]
        self.split_file.parent.mkdir(parents=True, exist_ok=True)

        self._dataset = None
        self._image_col = None
        self._label_col = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download (first time) or load from cache."""
        import os
        from datasets import load_dataset

        # Suppress Windows symlink warning (informational only, not an error)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        logger.info("Loading APTOS dataset from HuggingFace (cached at %s)...", self.cache_dir)
        try:
            self._dataset = load_dataset(
                self.ds_cfg["name"],
                cache_dir=str(self.cache_dir),
            )
        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise RuntimeError(
                f"Could not load dataset '{self.ds_cfg['name']}'. "
                f"Check your internet connection for the first download. Error: {e}"
            ) from e

        self._detect_columns()
        logger.info("Dataset loaded: %s", self._dataset)

    def inspect(self) -> Dict:
        """
        Phase 2: Full dataset inspection.
        Prints and returns structure, column names, sizes, class distribution.
        """
        if self._dataset is None:
            self.load()

        report = {}

        print("\n" + "=" * 70)
        print("  APTOS Dataset Inspection")
        print("=" * 70)

        # Structure
        print(f"\nDataset structure:\n  {self._dataset}")
        report["splits"] = list(self._dataset.keys())

        # Columns
        first_split = list(self._dataset.keys())[0]
        cols = self._dataset[first_split].column_names
        print(f"\nColumns: {cols}")
        print(f"  Image column : '{self._image_col}'")
        print(f"  Label column : '{self._label_col}'")
        report["image_col"] = self._image_col
        report["label_col"] = self._label_col
        report["columns"] = cols

        # Sizes
        print("\nSplit sizes:")
        report["sizes"] = {}
        total = 0
        for split, ds in self._dataset.items():
            n = len(ds)
            print(f"  {split:<12}: {n:>5} images")
            report["sizes"][split] = n
            total += n
        print(f"  {'TOTAL':<12}: {total:>5} images")
        report["total"] = total

        # Class distribution
        print("\nClass distribution:")
        report["class_distribution"] = {}
        for split, ds in self._dataset.items():
            labels = np.array([int(x) for x in ds[self._label_col]])
            dist = {}
            for cls in range(self.NUM_CLASSES):
                count = int(np.sum(labels == cls))
                pct = 100 * count / len(labels) if len(labels) > 0 else 0
                dist[cls] = {"count": count, "pct": round(pct, 1)}
                print(f"  [{split}] Grade {cls} ({self.CLASS_NAMES[cls]:<22}): {count:>5} ({pct:.1f}%)")
            report["class_distribution"][split] = dist

        # Verify labels are 0-4
        for split, ds in self._dataset.items():
            labels = np.array([int(x) for x in ds[self._label_col]])
            unique = sorted(np.unique(labels).tolist())
            assert all(0 <= int(l) <= 4 for l in unique), f"Unexpected labels in {split}: {unique}"
        print(f"\nLabels verified: all in range 0-{self.NUM_CLASSES - 1}")
        report["labels_valid"] = True

        # Sample inspection
        print("\nSample images (first 3):")
        first_ds = self._dataset[first_split]
        for i in range(min(3, len(first_ds))):
            sample = first_ds[i]
            img = sample[self._image_col]
            lbl = int(sample[self._label_col])
            print(f"  Sample {i}: label={lbl} ({self.CLASS_NAMES[lbl]}), "
                  f"image size={img.size}, mode={img.mode}")
        report["sample_ok"] = True

        # Corruption check (try accessing first 20 images)
        print("\nCorruption check (first 20 samples)...")
        bad = 0
        for i in range(min(20, len(first_ds))):
            try:
                img = first_ds[i][self._image_col]
                # HuggingFace already decodes images; test by converting to RGB
                _ = img.convert("RGB")
                arr = np.array(img)
                if arr.size == 0:
                    bad += 1
            except Exception as e:
                logger.warning("Sample %d unusable: %s", i, e)
                bad += 1
        print(f"  Unusable samples in first 20: {bad} / 20")
        report["corruption_in_first_20"] = bad
        print("=" * 70 + "\n")

        return report

    def get_splits(self, preprocessor=None) -> Tuple:
        """
        Returns train and validation PyTorch datasets.
        Uses the official split if available, otherwise creates a
        stratified 80/20 split with a fixed seed.
        """
        if self._dataset is None:
            self.load()

        # Check if the dataset already has train/validation splits
        has_train = "train" in self._dataset
        has_val = "validation" in self._dataset or "test" in self._dataset

        if has_train and has_val:
            logger.info("Using official dataset splits.")
            train_hf = self._dataset["train"]
            val_split = "validation" if "validation" in self._dataset else "test"
            val_hf = self._dataset[val_split]
            self._save_official_split_info(train_hf, val_hf)
        else:
            logger.info("No official split found. Creating stratified 80/20 split.")
            train_hf, val_hf = self._make_stratified_split()

        from dataset.aptos_dataset import APTOSDataset
        train_ds = APTOSDataset(train_hf, self._image_col, self._label_col,
                                preprocessor=preprocessor, training=True)
        val_ds = APTOSDataset(val_hf, self._image_col, self._label_col,
                               preprocessor=preprocessor, training=False)

        logger.info("Train: %d  |  Val: %d", len(train_ds), len(val_ds))
        return train_ds, val_ds

    def get_class_weights(self, train_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute inverse-frequency class weights for imbalance correction.
        strategy: 'balanced' (sklearn) or 'sqrt_inv'
        """
        from sklearn.utils.class_weight import compute_class_weight

        if train_labels is None:
            if self._dataset is None:
                self.load()
            split = "train" if "train" in self._dataset else list(self._dataset.keys())[0]
            train_labels = np.array([int(x) for x in self._dataset[split][self._label_col]])

        strategy = self.cfg.get("training", {}).get("class_weighting", "balanced")

        if strategy == "none":
            weights = np.ones(self.NUM_CLASSES)
        elif strategy == "sqrt_inv":
            counts = np.bincount(train_labels, minlength=self.NUM_CLASSES).astype(float)
            counts = np.maximum(counts, 1)
            weights = 1.0 / np.sqrt(counts)
            weights = weights / weights.sum() * self.NUM_CLASSES
        else:  # 'balanced'
            weights = compute_class_weight(
                class_weight="balanced",
                classes=np.arange(self.NUM_CLASSES),
                y=train_labels,
            )

        logger.info("Class weights (%s): %s", strategy,
                    [f"{w:.3f}" for w in weights])
        return weights.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_columns(self):
        """Auto-detect the image and label column names."""
        first_split = list(self._dataset.keys())[0]
        cols = self._dataset[first_split].column_names

        # Image column
        img_candidates = ["image", "img", "Image", "retinal_image", "fundus"]
        self._image_col = next((c for c in img_candidates if c in cols), None)
        if self._image_col is None:
            features = self._dataset[first_split].features
            for col, feat in features.items():
                if "Image" in type(feat).__name__:
                    self._image_col = col
                    break
        if self._image_col is None:
            raise ValueError(f"Cannot detect image column. Available: {cols}")

        # Label column — prefer integer (label_code) over string (label)
        # bumbledeep/aptos has both: 'label_code' (int) and 'label' (string)
        int_candidates = ["label_code", "grade", "dr_grade", "diagnosis_int",
                          "label_int", "target"]
        str_candidates = ["label", "labels", "diagnosis", "Label", "Diagnosis"]
        all_candidates = int_candidates + str_candidates

        self._label_col = next(
            (c for c in all_candidates if c in cols and c != self._image_col),
            None
        )
        if self._label_col is None:
            # Fallback: any non-image column
            for col in cols:
                if col != self._image_col:
                    self._label_col = col
                    break
        if self._label_col is None:
            raise ValueError(f"Cannot detect label column. Available: {cols}")

        # Check if the detected label column contains integers; if string → remap
        sample = self._dataset[first_split][0]
        lbl_val = sample[self._label_col]
        if isinstance(lbl_val, str):
            logger.warning(
                "Label column '%s' contains strings — building label-to-int mapping.",
                self._label_col
            )
            self._build_string_label_map()

        logger.info("Auto-detected columns: image='%s', label='%s'",
                    self._image_col, self._label_col)

    def _build_string_label_map(self):
        """
        For datasets where labels are strings (e.g. 'moderate_retinopathy'),
        build a string→int mapping. The integer ordering follows ICDR 0-4.
        """
        LABEL_MAP = {
            "no_diabetic_retinopathy": 0,
            "no_dr": 0,
            "no dr": 0,
            "mild_retinopathy": 1,
            "mild dr": 1,
            "mild_dr": 1,
            "moderate_retinopathy": 2,
            "moderate dr": 2,
            "moderate_dr": 2,
            "severe_retinopathy": 3,
            "severe dr": 3,
            "severe_dr": 3,
            "proliferative_retinopathy": 4,
            "proliferative dr": 4,
            "proliferative_dr": 4,
        }
        self._str_label_map = LABEL_MAP
        logger.info("String label map built with %d entries.", len(LABEL_MAP))

    def _str_to_int_label(self, lbl) -> int:
        """Convert a string label to integer 0-4."""
        if hasattr(self, "_str_label_map") and isinstance(lbl, str):
            key = lbl.lower().strip()
            if key in self._str_label_map:
                return self._str_label_map[key]
        return int(lbl)

    def _make_stratified_split(self):
        """Create reproducible stratified 80/20 split."""
        from sklearn.model_selection import train_test_split

        first_split = list(self._dataset.keys())[0]
        full_ds = self._dataset[first_split]
        labels = np.array([int(x) for x in full_ds[self._label_col]])
        indices = np.arange(len(full_ds))

        val_frac = self.ds_cfg.get("val_fraction", 0.20)
        seed = self.ds_cfg.get("split_seed", 42)

        train_idx, val_idx = train_test_split(
            indices, test_size=val_frac, random_state=seed, stratify=labels
        )

        # Save split for reproducibility
        split_info = {
            "strategy": "stratified_80_20",
            "seed": seed,
            "val_fraction": val_frac,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        }
        with open(self.split_file, "w") as f:
            json.dump(split_info, f, indent=2)
        logger.info("Split saved to %s", self.split_file)

        train_hf = full_ds.select(train_idx.tolist())
        val_hf = full_ds.select(val_idx.tolist())
        return train_hf, val_hf

    def _save_official_split_info(self, train_hf, val_hf):
        split_info = {
            "strategy": "official_split",
            "train_size": len(train_hf),
            "val_size": len(val_hf),
        }
        with open(self.split_file, "w") as f:
            json.dump(split_info, f, indent=2)
