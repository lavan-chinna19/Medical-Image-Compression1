"""Small U-Net segmentation architecture with Monte Carlo dropout and evaluation metrics.

Architecture:
- 4 down/up levels with base 32 channels (32 -> 64 -> 128 -> 256 -> 512 bottleneck).
- DoubleConv blocks: Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> ReLU.
- Spatial dropout (Dropout2d(0.1)) in the decoder stages to enable Monte Carlo dropout uncertainty.
- Output: 1 channel of unnormalized logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two sequential 3x3 convolutions, each with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small 4-level U-Net for medical image segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        b = base_channels

        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, b)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(b * 2, b * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(b * 4, b * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(b * 8, b * 16)

        # Decoder (Upsampling) with Dropout2d for Monte Carlo uncertainty
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.drop4 = nn.Dropout2d(p=dropout_p)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.drop3 = nn.Dropout2d(p=dropout_p)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.drop2 = nn.Dropout2d(p=dropout_p)

        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.drop1 = nn.Dropout2d(p=dropout_p)

        # Final projection to 1 channel of logits
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections and spatial dropout
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.drop4(self.dec4(d4))

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.drop3(self.dec3(d3))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.drop2(self.dec2(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.drop1(self.dec1(d1))

        return self.out_conv(d1)


def enable_mc_dropout(model: nn.Module) -> None:
    """Set dropout layers to train mode while keeping BatchNorm in eval mode."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute soft or binary Dice coefficient over batch."""
    p_flat = pred.contiguous().view(pred.shape[0], -1)
    t_flat = target.contiguous().view(target.shape[0], -1)

    intersection = (p_flat * t_flat).sum(dim=1)
    cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return dice.mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Intersection-over-Union (Jaccard Index) over batch."""
    p_flat = pred.contiguous().view(pred.shape[0], -1)
    t_flat = target.contiguous().view(target.shape[0], -1)

    intersection = (p_flat * t_flat).sum(dim=1)
    union = p_flat.sum(dim=1) + t_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined Binary Cross Entropy and soft Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dice = 1.0 - dice_score(probs, target)
    return bce + dice
