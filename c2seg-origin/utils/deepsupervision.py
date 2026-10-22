# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================
# Loss Functions
# ================================================
class AsymmetricDice(nn.Module):
    """
    Asymmetric Dice Loss with weighting for positive and negative pixels.
    """
    def __init__(self, weights=(0.4, 0.6)):
        super().__init__()
        self.w_neg, self.w_pos = weights

    def forward(self, targets: torch.Tensor, logits: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        batch_size = logits.size(0)
        targets = targets.view(batch_size, -1)
        logits = logits.view(batch_size, -1)

        weights = targets.detach() * (self.w_pos - self.w_neg) + self.w_neg
        t, p = weights * targets, weights * logits

        intersection = (t * p).sum(-1)
        union = (t * t).sum(-1) + (p * p).sum(-1)
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        return dice_loss.mean()


class AsymmetricBCE(nn.Module):
    """
    Asymmetric Binary Cross-Entropy Loss with different weights
    for positive and negative samples.
    """
    def __init__(self, weights=(0.4, 0.6)):
        super().__init__()
        self.w_neg, self.w_pos = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.flatten()
        targets = targets.flatten()
        loss = F.binary_cross_entropy(logits, targets, reduction="none")

        pos_mask = (targets > 0.5).float()
        neg_mask = (targets < 0.5).float()
        pos_weight = pos_mask.sum().clamp_min(1e-12)
        neg_weight = neg_mask.sum().clamp_min(1e-12)

        weighted_loss = (self.w_pos * pos_mask * loss / pos_weight +
                         self.w_neg * neg_mask * loss / neg_weight).sum()
        return weighted_loss


class AsymmetricDiceBCE(nn.Module):
    """
    Combination of Asymmetric BCE and Asymmetric Dice Loss.
    """
    def __init__(self, weights=(0.4, 0.6), alpha: float = 0.5):
        super().__init__()
        self.dice = AsymmetricDice(weights)
        self.bce = AsymmetricBCE(weights)
        self.alpha = alpha  # balance factor between BCE and Dice

    def forward(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(targets, logits)
        bce_loss = self.bce(logits, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


# ================================================
# PatchMask Generator
# ================================================
class PatchMaskGenerator(nn.Module):
    """
    Generates binary patch-level masks by dividing the image into
    non-overlapping patches and marking a patch as 1 if it contains
    any foreground pixels.
    """
    def __init__(self, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.conv.weight.data.fill_(1.0)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_mean = self.conv(x.float()) / (self.patch_size * self.patch_size)
        return (patch_mean > 0).float()

