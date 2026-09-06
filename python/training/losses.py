"""
RetinaGuard — Advanced Loss Functions for Imbalanced & Ordinal DR Classification
================================================================================
Implements:
1. ClassWeightedCrossEntropy
2. FocalLoss (multi-class with alpha weighting and gamma focusing)
3. ClassBalancedFocalLoss (Cui et al. effective number of samples)
4. OrdinalFocalLoss (Focal Loss + Expected Grade Distance penalty for QWK optimization)

All loss functions are fully differentiable and compatible with PyTorch.
"""

import math
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples and focuses training on hard minority classes (e.g. Grade 4).
    """
    def __init__(self, alpha: Optional[Union[torch.Tensor, list, np.ndarray]] = None,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C) raw unnormalized scores
        targets: (B,) class labels in [0, C-1]
        """
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Gather probability of true class
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al., CVPR 2019):
    Computes effective number of samples: E_n = (1 - beta^n) / (1 - beta)
    Weights: alpha_i = (1 - beta) / (1 - beta^{n_i})
    """
    def __init__(self, samples_per_class: list, beta: float = 0.999,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        effective_num = [1.0 - (beta ** n) for n in samples_per_class]
        weights = [(1.0 - beta) / max(en, 1e-8) for en in effective_num]
        total_w = sum(weights)
        norm_weights = [w / total_w * len(samples_per_class) for w in weights]
        
        self.register_buffer("alpha", torch.tensor(norm_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        alpha_t = self.alpha[targets]
        loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalFocalLoss(nn.Module):
    """
    Combined Focal Loss + Ordinal Distance Penalty.
    
    Since Diabetic Retinopathy severity is an ordinal progression (0 < 1 < 2 < 3 < 4),
    this objective penalizes:
    1. Classification error via Focal Loss (focused on hard minority examples)
    2. Distance between expected continuous grade and true grade:
       E[y] = sum_{c=0}^4 c * p_c
       L_ordinal = (E[y] - y)^2
       
    This directly aligns with Quadratic Weighted Kappa (QWK), which penalizes
    the square of the ordinal distance between true and predicted grades.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 ordinal_lambda: float = 0.3, num_classes: int = 5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.ordinal_lambda = ordinal_lambda
        self.num_classes = num_classes
        self.register_buffer("class_indices", torch.arange(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)

        # Compute expected grade
        probs = F.softmax(logits, dim=1)  # (B, C)
        expected_grade = (probs * self.class_indices).sum(dim=1)  # (B,)
        
        # Ordinal distance loss (MSE between expected grade and target grade)
        target_floats = targets.float()
        ordinal_loss = F.mse_loss(expected_grade, target_floats)

        return focal_loss + self.ordinal_lambda * ordinal_loss


def build_loss_function(loss_name: str, class_weights: Optional[torch.Tensor] = None,
                        train_counts: Optional[list] = None, gamma: float = 2.0,
                        ordinal_lambda: float = 0.25) -> nn.Module:
    """Factory function for loss modules."""
    name = loss_name.lower().strip()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    elif name == "cb_focal":
        assert train_counts is not None, "train_counts required for CB-Focal"
        return ClassBalancedFocalLoss(samples_per_class=train_counts, gamma=gamma)
    elif name == "ordinal_focal":
        return OrdinalFocalLoss(alpha=class_weights, gamma=gamma, ordinal_lambda=ordinal_lambda)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
