"""
RetinaGuard — APTOS PyTorch Dataset Wrapper
============================================
Wraps a HuggingFace dataset split into a PyTorch Dataset that applies
the retinal preprocessing pipeline and augmentation.
"""

from typing import Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class APTOSDataset(Dataset):
    """
    PyTorch Dataset that wraps a HuggingFace APTOS dataset split.

    Args:
        hf_dataset   : HuggingFace dataset split
        image_col    : Name of the image column
        label_col    : Name of the label column
        preprocessor : RetinalPreprocessor instance (applies transforms)
        training     : If True, apply augmentation; else, apply val transforms only
    """

    CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

    def __init__(self, hf_dataset, image_col: str, label_col: str,
                 preprocessor=None, training: bool = False):
        self.hf_dataset = hf_dataset
        self.image_col = image_col
        self.label_col = label_col
        self.preprocessor = preprocessor
        self.training = training

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        sample = self.hf_dataset[idx]
        img: Image.Image = sample[self.image_col]
        raw_label = sample[self.label_col]
        # Handle both integer and string label columns
        if isinstance(raw_label, str):
            # String label (e.g. 'moderate_retinopathy') — convert via mapping
            STRING_MAP = {
                "no_diabetic_retinopathy": 0, "no_dr": 0, "no dr": 0,
                "mild_retinopathy": 1, "mild_dr": 1, "mild dr": 1,
                "moderate_retinopathy": 2, "moderate_dr": 2, "moderate dr": 2,
                "severe_retinopathy": 3, "severe_dr": 3, "severe dr": 3,
                "proliferative_retinopathy": 4, "proliferative_dr": 4, "proliferative dr": 4,
            }
            label = STRING_MAP.get(raw_label.lower().strip(), int(raw_label))
        else:
            label = int(raw_label)

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply preprocessing + augmentation
        if self.preprocessor is not None:
            if self.training:
                tensor = self.preprocessor.transform_train(img)
            else:
                tensor = self.preprocessor.transform_val(img)
        else:
            # Fallback: plain resize + tensor
            import torchvision.transforms as T
            tensor = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(img)

        return tensor, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> np.ndarray:
        """Return all labels as numpy array (for computing class weights etc.)."""
        raw = self.hf_dataset[self.label_col]
        return np.array([int(x) for x in raw])
