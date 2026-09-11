"""
unet.py
=======
Dual-Head U-Net Architecture for Retinal Fundus Image Segmentation.

Features:
- Shared Encoder & Decoder backbone (lightweight 4-level U-Net).
- Decoupled Heads:
  1. lesion_head: 4 channels (MA, HE, EX, SE)
  2. anatomy_head: 1 channel (Optic Disc)
- Supports ONNX export with unified 5-channel sigmoid output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if sizes differ slightly
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DualHeadUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_lesion_classes: int = 4,
        num_anatomy_classes: int = 1,
        base_features: int = 32,
        export_mode: bool = False
    ):
        super().__init__()
        self.export_mode = export_mode

        # Encoder
        self.inc = DoubleConv(in_channels, base_features)              # 32
        self.down1 = Down(base_features, base_features * 2)             # 64
        self.down2 = Down(base_features * 2, base_features * 4)         # 128
        self.down3 = Down(base_features * 4, base_features * 8)         # 256
        self.down4 = Down(base_features * 8, base_features * 8)         # 256 (bottleneck)

        # Decoder
        self.up1 = Up(base_features * 8 + base_features * 8, base_features * 4)   # 256 + 256 -> 128
        self.up2 = Up(base_features * 4 + base_features * 4, base_features * 2)   # 128 + 128 -> 64
        self.up3 = Up(base_features * 2 + base_features * 2, base_features)       # 64 + 64 -> 32
        self.up4 = Up(base_features + base_features, base_features)               # 32 + 32 -> 32

        # Distinct Decoupled Task Heads
        self.lesion_head = nn.Conv2d(base_features, num_lesion_classes, kernel_size=1)
        self.anatomy_head = nn.Conv2d(base_features, num_anatomy_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Feature extraction
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Shared feature reconstruction
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        feats = self.up4(x, x1)

        # Heads
        lesion_logits = self.lesion_head(feats)
        anatomy_logits = self.anatomy_head(feats)

        if self.export_mode:
            # Concatenate along channel dim: [B, 5, H, W] -> Sigmoid
            combined = torch.cat([lesion_logits, anatomy_logits], dim=1)
            return torch.sigmoid(combined)
        
        return lesion_logits, anatomy_logits
