"""
RetinaGuard — Retinal Image Preprocessor
==========================================
Implements a clinically appropriate preprocessing pipeline for retinal fundus images.

The SAME preprocessing applied during training MUST be applied during inference.
This module is used by:
  - training/train.py       (training + validation transforms)
  - inference/predict.py    (inference transforms)

Design:
  - Training : augmentation + normalization
  - Validation/Inference : deterministic crop + normalization (no augmentation)

Usage:
    from preprocessing.retinal_preprocessor import RetinalPreprocessor
    prep = RetinalPreprocessor(cfg)
    tensor = prep.transform_val(pil_image)          # inference
    tensor = prep.transform_train(pil_image)        # training
    enhanced_pil = prep.enhance_for_display(pil_image)  # UI display
"""

import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


class RetinalPreprocessor:
    """
    Preprocessing and augmentation pipeline for retinal fundus images.

    Key design decisions:
    - CLAHE applied before converting to tensor (OpenCV/PIL domain)
    - ImageNet normalization after CLAHE, as the backbone expects it
    - Augmentation applied to training set only
    - Identical val/inference transforms for consistency
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prep_cfg = cfg.get("preprocessing", {})
        self.aug_cfg = cfg.get("augmentation", {})

        self.image_size = int(self.prep_cfg.get("image_size", 224))
        self.mean = self.prep_cfg.get("mean", [0.485, 0.456, 0.406])
        self.std = self.prep_cfg.get("std", [0.229, 0.224, 0.225])

        # CLAHE parameters
        self.clahe_clip = float(self.prep_cfg.get("clahe_clip_limit", 2.0))
        self.clahe_tile = int(self.prep_cfg.get("clahe_tile_grid", 8))

        # Build transforms
        self.transform_train = self._build_train_transform()
        self.transform_val = self._build_val_transform()

    # ──────────────────────────────────────────────────────────────────────────
    # Transform builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_train_transform(self) -> T.Compose:
        """Training transform: CLAHE → augmentation → normalize."""
        aug = self.aug_cfg
        transforms = [
            T.Lambda(self._apply_clahe),                     # CLAHE enhancement
            T.Resize((self.image_size + 32, self.image_size + 32)),  # slight over-size
        ]

        if aug.get("horizontal_flip", True):
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        if aug.get("vertical_flip", False):
            transforms.append(T.RandomVerticalFlip(p=0.3))

        rotation = aug.get("rotation_degrees", 15)
        scale = aug.get("random_crop_scale", [0.85, 1.0])
        transforms.append(
            T.RandomResizedCrop(
                self.image_size,
                scale=tuple(scale),
                ratio=(0.9, 1.1),
            )
        )
        if rotation > 0:
            transforms.append(T.RandomRotation(degrees=rotation))

        brightness = aug.get("brightness_jitter", 0.2)
        contrast = aug.get("contrast_jitter", 0.2)
        saturation = aug.get("saturation_jitter", 0.1)
        if brightness or contrast or saturation:
            transforms.append(
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=0.0,   # Never jitter hue — retinal colour is clinically meaningful
                )
            )

        transforms += [
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        return T.Compose(transforms)

    def _build_val_transform(self) -> T.Compose:
        """Validation/inference transform: CLAHE → deterministic resize → normalize."""
        return T.Compose([
            T.Lambda(self._apply_clahe),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # CLAHE preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_clahe(self, img: Image.Image) -> Image.Image:
        """
        Apply CLAHE to the L channel in LAB colour space.
        Enhances local contrast without clipping global brightness.
        Falls back gracefully if OpenCV is unavailable.
        """
        try:
            import cv2

            img_np = np.array(img.convert("RGB"))
            # Convert to LAB
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab)

            # Apply CLAHE to L channel only
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_tile, self.clahe_tile),
            )
            l_clahe = clahe.apply(l_channel)

            # Merge back and convert to RGB
            lab_clahe = cv2.merge([l_clahe, a, b])
            rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            return Image.fromarray(rgb_clahe)

        except ImportError:
            # OpenCV not available — fall back to PIL contrast enhancement
            logger.debug("OpenCV not available; using PIL contrast enhancement instead of CLAHE.")
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1.3)
        except Exception as e:
            logger.debug("CLAHE failed (%s); returning original image.", e)
            return img

    # ──────────────────────────────────────────────────────────────────────────
    # Display helpers (for UI — not used in training)
    # ──────────────────────────────────────────────────────────────────────────

    def enhance_for_display(self, img: Image.Image) -> Image.Image:
        """
        Produce a visually enhanced version for the UI overlay.
        NOT used during training or inference — for display only.
        """
        enhanced = self._apply_clahe(img.convert("RGB"))
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1.05)
        return enhanced

    def tensor_to_display_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert a normalized inference tensor back to a displayable PIL image.
        Reverses ImageNet normalization.
        """
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        img_t = tensor.cpu().clone() * std + mean
        img_t = img_t.clamp(0, 1)
        return T.ToPILImage()(img_t)

    def preprocess_for_inference(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess a single PIL image for model inference.
        Returns a (1, 3, H, W) tensor.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.transform_val(img)
        return tensor.unsqueeze(0)  # Add batch dimension
