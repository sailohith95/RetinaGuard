"""
dataset.py
==========
PyTorch Dataset for IDRiD Retinal Fundus Lesion and Anatomical Segmentation.

Strictly decouples:
- Lesion Pathology (4 channels: MA, HE, EX, SE)
- Normal Anatomy (1 channel: Optic Disc)
"""

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

LESION_CLASSES = [
    ("1. Microaneurysms", "MA"),
    ("2. Haemorrhages", "HE"),
    ("3. Hard Exudates", "EX"),
    ("4. Soft Exudates", "SE"),
]

ANATOMY_CLASSES = [
    ("5. Optic Disc", "OD"),
]


class IDRiDSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False
    ):
        """
        Args:
            root_dir: Path to 'data/idrid/A. Segmentation'
            split: 'train' (54 images) or 'test' (27 images)
            image_size: target square dimension (default: 256x256)
            augment: apply random flips and rotations if True
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        subfolder = "a. Training Set" if split == "train" else "b. Testing Set"
        self.img_dir = self.root_dir / "1. Original Images" / subfolder
        self.gt_dir = self.root_dir / "2. All Segmentation Groundtruths" / subfolder

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        self.image_files = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        img_path = self.image_files[idx]
        img_id = img_path.stem

        # Load original image in RGB
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Resize image
        resized_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(resized_img)
        # Normalize with standard ImageNet statistics
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Load 4 lesion masks (MA, HE, EX, SE)
        lesion_masks_np = []
        for folder, abbrev in LESION_CLASSES:
            mask_path = self.gt_dir / folder / f"{img_id}_{abbrev}.tif"
            if mask_path.exists():
                mask_pil = Image.open(mask_path).convert("L")
                mask_resized = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
                mask_arr = (np.array(mask_resized) > 0).astype(np.float32)
            else:
                mask_arr = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            lesion_masks_np.append(mask_arr)

        lesion_tensor = torch.from_numpy(np.stack(lesion_masks_np, axis=0))  # [4, H, W]

        # Load 1 anatomy mask (Optic Disc)
        od_path = self.gt_dir / "5. Optic Disc" / f"{img_id}_OD.tif"
        if od_path.exists():
            od_pil = Image.open(od_path).convert("L")
            od_resized = od_pil.resize((self.image_size, self.image_size), Image.NEAREST)
            od_arr = (np.array(od_resized) > 0).astype(np.float32)
        else:
            od_arr = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        anatomy_tensor = torch.from_numpy(od_arr).unsqueeze(0)  # [1, H, W]

        # Data augmentation for training
        if self.augment:
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                lesion_tensor = TF.hflip(lesion_tensor)
                anatomy_tensor = TF.hflip(anatomy_tensor)
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                lesion_tensor = TF.vflip(lesion_tensor)
                anatomy_tensor = TF.vflip(anatomy_tensor)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img_tensor = TF.rotate(img_tensor, angle)
                lesion_tensor = TF.rotate(lesion_tensor, angle)
                anatomy_tensor = TF.rotate(anatomy_tensor, angle)

        meta = {
            "image_id": img_id,
            "filename": img_path.name,
            "orig_size": (orig_w, orig_h)
        }

        return img_tensor, lesion_tensor, anatomy_tensor, meta
