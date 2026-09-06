"""
RetinaGuard — EfficientNet-B0 DR Classifier
=============================================
Transfer learning model for 5-class diabetic retinopathy grading.

Architecture:
  - EfficientNet-B0 pretrained on ImageNet
  - Custom classification head: Dropout → Linear(5)
  - Two-stage training: frozen backbone → fine-tune upper layers

The model is configurable to allow swapping to ResNet50, EfficientNet-B2, etc.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Supported architectures
SUPPORTED_ARCHITECTURES = {
    "efficientnet_b0",
    "efficientnet_b2",
    "resnet50",
    "resnet34",
}


class DRClassifier(nn.Module):
    """
    DR severity classifier built on a pretrained ImageNet backbone.

    Args:
        architecture : model name (e.g. 'efficientnet_b0')
        num_classes  : 5 for ICDR DR grading (must NOT change)
        pretrained   : whether to use ImageNet pretrained weights
        dropout      : dropout rate in classification head
    """

    NUM_CLASSES = 5

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert num_classes == 5, "DR classifier must have exactly 5 output classes (ICDR 0-4)"
        assert architecture in SUPPORTED_ARCHITECTURES, (
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {SUPPORTED_ARCHITECTURES}"
        )

        self.architecture = architecture
        self.num_classes = num_classes
        self.dropout = dropout

        # Build backbone + custom head
        self.backbone, in_features = self._build_backbone(architecture, pretrained)
        self.classifier = self._build_head(in_features, num_classes, dropout)

        logger.info(
            "DRClassifier: arch=%s, pretrained=%s, in_features=%d, dropout=%.2f",
            architecture, pretrained, in_features, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (apply softmax externally for probabilities)."""
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Stage 1: Freeze all backbone parameters, train head only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen. Training classification head only.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None):
        """
        Stage 2: Unfreeze backbone for fine-tuning.
        If unfreeze_last_n_blocks is set, only the last N blocks are unfrozen.
        """
        if unfreeze_last_n_blocks is None:
            # Unfreeze everything
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen for fine-tuning.")
        else:
            # Unfreeze last N blocks of EfficientNet features
            try:
                if hasattr(self.backbone, "features"):
                    features_mod = self.backbone.features
                elif isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0 and isinstance(self.backbone[0], nn.Sequential):
                    features_mod = self.backbone[0]
                else:
                    raise AttributeError("Cannot find feature extraction blocks in backbone")

                blocks = list(features_mod.children())
                freeze_until = max(0, len(blocks) - unfreeze_last_n_blocks)
                for i, block in enumerate(blocks):
                    for param in block.parameters():
                        param.requires_grad = (i >= freeze_until)
                logger.info(
                    "Unfroze last %d of %d backbone blocks (blocks %d to %d unfrozen).",
                    unfreeze_last_n_blocks, len(blocks), freeze_until, len(blocks) - 1
                )
            except AttributeError:
                # For ResNet or other architectures
                for param in self.backbone.parameters():
                    param.requires_grad = True
                logger.info("Full backbone unfrozen (non-EfficientNet fallback).")

    def get_trainable_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal builders
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_backbone(architecture: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Return (backbone_without_head, in_features_for_head)."""
        from torchvision import models

        weights_arg = "DEFAULT" if pretrained else None

        if architecture == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights_arg)
            in_features = base.classifier[1].in_features
            # Remove the classifier — we'll replace it
            backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())

        elif architecture == "efficientnet_b2":
            base = models.efficientnet_b2(weights=weights_arg)
            in_features = base.classifier[1].in_features
            backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())

        elif architecture == "resnet50":
            base = models.resnet50(weights=weights_arg)
            in_features = base.fc.in_features
            # Remove fc layer
            backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())

        elif architecture == "resnet34":
            base = models.resnet34(weights=weights_arg)
            in_features = base.fc.in_features
            backbone = nn.Sequential(*list(base.children())[:-1], nn.Flatten())

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        return backbone, in_features

    @staticmethod
    def _build_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
        """Classification head: Dropout → Linear → (softmax applied externally)."""
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*layers)


def build_model(cfg: dict) -> DRClassifier:
    """Factory function — builds model from config dict."""
    model_cfg = cfg.get("model", {})
    model = DRClassifier(
        architecture=model_cfg.get("architecture", "efficientnet_b0"),
        num_classes=5,
        pretrained=model_cfg.get("pretrained", True),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )

    params = model.get_trainable_params()
    logger.info(
        "Model built: %d total params, %d trainable (%.1f%% frozen)",
        params["total"], params["trainable"],
        100 * params["frozen"] / max(params["total"], 1),
    )
    return model
