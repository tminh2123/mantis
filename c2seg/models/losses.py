"""
Losses for C2Seg (matching c2seg-origin implementation):
  - CaCLLoss: Cluster-aware Contrastive Learning loss (Stage I)
  - WeightedBCE: Asymmetric Binary Cross-Entropy Loss
  - WeightedDiceLoss: Asymmetric Dice Loss
  - WeightedDiceBCE: Combined Weighted Dice + BCE (main segmentation loss)
  - PatchMaskGenerator: Generates patch-level supervision masks
  - DeepSupervisionLoss: Multi-scale loss with patch masks (Stage II)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaCLLoss(nn.Module):
    """
    Cluster-aware Contrastive Learning loss.
    Converts intra-batch text similarity into soft labels to suppress false negatives.

    Args:
        temperature: tau for cross-modal logits (default 0.07)
        eps: soft-label mixing weight (default 0.8)
    """

    def __init__(self, temperature: float = 0.07, eps: float = 0.8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, v: torch.Tensor, l: torch.Tensor, l_raw: torch.Tensor):
        """
        v:     [B, d]  L2-normalised image embeddings
        l:     [B, d]  L2-normalised sentence embeddings (projected)
        l_raw: [B, d]  L2-normalised sentence embeddings from frozen encoder
                       (used to build soft-label matrix M)
        """
        B = v.shape[0]

        # --- build soft labels from text-text similarity ---
        M = torch.mm(l_raw, l_raw.t())          # [B, B] cosine similarity

        # row-mean debias + clip negatives
        mu = M.mean(dim=1, keepdim=True)
        M_deb = (M - mu).clamp(min=0)           # [B, B]

        # soft targets via temperature softmax
        Y_hat = F.softmax(M_deb / self.tau, dim=1)   # [B, B]

        # mix with hard self-positive
        I = torch.eye(B, device=v.device)
        Y = self.eps * Y_hat + (1 - self.eps) * I    # [B, B]

        # --- cross-modal logits ---
        S = torch.mm(v, l.t()) / self.tau            # [B, B]

        P_v2l = F.log_softmax(S,   dim=1)            # image -> text
        P_l2v = F.log_softmax(S.t(), dim=1)          # text  -> image

        loss = -(Y * P_v2l).sum(dim=1).mean() \
               -(Y.t() * P_l2v).sum(dim=1).mean()
        return loss * 0.5


# ================================================
# Asymmetric Loss Functions (from c2seg-origin)
# ================================================

class WeightedBCE(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    Applies different weights to positive and negative samples.
    
    Args:
        weights: tuple (neg_weight, pos_weight), default (0.4, 0.6)
    """
    def __init__(self, weights=(0.4, 0.6)):
        super().__init__()
        self.w_neg, self.w_pos = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        
        loss = F.binary_cross_entropy(logits, targets, reduction='none')
        
        pos_mask = (targets > 0.5).float()
        neg_mask = (targets < 0.5).float()
        pos_weight = pos_mask.sum().clamp_min(1e-12)
        neg_weight = neg_mask.sum().clamp_min(1e-12)

        weighted_loss = (self.w_pos * pos_mask * loss / pos_weight +
                         self.w_neg * neg_mask * loss / neg_weight).sum()
        return weighted_loss


class WeightedDiceLoss(nn.Module):
    """
    Weighted Dice Loss.
    Applies different weights to positive and negative pixels.
    
    Args:
        weights: tuple (neg_weight, pos_weight), default (0.4, 0.6)
    """
    def __init__(self, weights=(0.4, 0.6)):
        super().__init__()
        self.w_neg, self.w_pos = weights

    def forward(self, targets: torch.Tensor, logits: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        batch_size = logits.size(0)
        targets = targets.view(batch_size, -1)
        logits = logits.view(batch_size, -1)

        # Weight pixels based on class
        weights = targets.detach() * (self.w_pos - self.w_neg) + self.w_neg
        t = weights * targets
        p = weights * logits

        intersection = (t * p).sum(-1)
        union = (t * t).sum(-1) + (p * p).sum(-1)
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        return dice_loss.mean()


class WeightedDiceBCE(nn.Module):
    """
    Combined Weighted Dice + Binary Cross-Entropy Loss.
    This is the main segmentation loss used in c2seg-origin.
    
    Args:
        weights: tuple (neg_weight, pos_weight) for asymmetric weighting
    """
    def __init__(self, weights=(0.4, 0.6)):
        super().__init__()
        self.dice_loss = WeightedDiceLoss(weights)
        self.bce_loss = WeightedBCE(weights)

    def forward(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            targets: [B, 1, H, W] or [B, H, W] ground truth mask
            logits: [B, 1, H, W] or [B, H, W] predicted probabilities (after sigmoid)
        """
        dice = self.dice_loss(targets, logits)
        bce = self.bce_loss(logits, targets)
        return 0.5 * dice + 0.5 * bce


class AsymmetricDiceBCE(nn.Module):
    """
    Alternative naming for WeightedDiceBCE (alias for compatibility).
    """
    def __init__(self, weights=(0.4, 0.6), alpha: float = 0.5):
        super().__init__()
        self.dice = WeightedDiceLoss(weights)
        self.bce = WeightedBCE(weights)
        self.alpha = alpha

    def forward(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(targets, logits)
        bce_loss = self.bce(logits, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


# ================================================
# Patch Mask Generator (from c2seg-origin)
# ================================================

class PatchMaskGenerator(nn.Module):
    """
    Generates binary patch-level masks by dividing the image into
    non-overlapping patches and marking a patch as 1 if it contains
    any foreground pixels.
    
    Used for multi-scale deep supervision at different patch sizes.
    
    Args:
        patch_size: size of each patch (e.g., 2, 4, 8, 16)
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
        # Fill with ones to compute sum of pixels in each patch
        self.conv.weight.data.fill_(1.0)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] binary mask
        Returns:
            [B, 1, H/patch_size, W/patch_size] patch-level mask
        """
        # Move conv to same device as input
        self.conv = self.conv.to(x.device)
        patch_mean = self.conv(x.float()) / (self.patch_size * self.patch_size)
        return (patch_mean > 0).float()


# ================================================
# Deep Supervision Loss (matching c2seg-origin)
# ================================================

class DeepSupervisionLoss(nn.Module):
    """
    Multi-scale deep supervision loss matching c2seg-origin.
    
    Uses different asymmetric weights at each scale:
      - Scale 0 (full res): weights [0.5, 0.5]
      - Scale 1 (patch 2):  weights [0.4, 0.6]
      - Scale 2 (patch 4):  weights [0.3, 0.7]
      - Scale 3 (patch 8):  weights [0.2, 0.8]
      - Scale 4 (patch 16): weights [0.1, 0.9]
    
    Loss weights: [0.5, 0.2, 0.15, 0.1, 0.05]
    
    Args:
        loss_weights: list of weights for each scale (default from config)
        asymmetric_weights: list of (neg, pos) weights for each scale
    """

    def __init__(
        self, 
        loss_weights: list = None,
        asymmetric_weights: list = None,
        patch_sizes: list = None
    ):
        super().__init__()
        
        # Default loss weights from c2seg-origin config
        self.loss_weights = loss_weights or [0.5, 0.2, 0.15, 0.1, 0.05]
        
        # Default asymmetric weights from c2seg-origin train_model.py
        self.asymmetric_weights = asymmetric_weights or [
            (0.5, 0.5),   # Scale 0: balanced
            (0.4, 0.6),   # Scale 1: slight positive bias
            (0.3, 0.7),   # Scale 2: more positive bias
            (0.2, 0.8),   # Scale 3: strong positive bias
            (0.1, 0.9),   # Scale 4: very strong positive bias
        ]
        
        # Patch sizes for generating patch-level masks
        self.patch_sizes = patch_sizes or [2, 4, 8, 16]
        
        # Create loss functions for each scale
        self.criterions = nn.ModuleList([
            WeightedDiceBCE(w) for w in self.asymmetric_weights
        ])
        
        # Create patch mask generators (for scales 1-4)
        self.patch_gens = nn.ModuleList([
            PatchMaskGenerator(ps) for ps in self.patch_sizes
        ])

    def forward(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: tuple of (out0, out1, out2, out3, out4) at different scales
                     out0: [B, 1, H, W]     - full resolution
                     out1: [B, 1, H/2, W/2]
                     out2: [B, 1, H/4, W/4]
                     out3: [B, 1, H/8, W/8]
                     out4: [B, 1, H/16, W/16]
            targets: [B, 1, H, W] binary mask at full resolution
            
        Returns:
            Weighted sum of losses at all scales
        """
        # Ensure targets has channel dimension
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Generate patch masks for scales 1-4
        patch_masks = [gen(targets) for gen in self.patch_gens]
        
        # All masks: [full_res, patch2, patch4, patch8, patch16]
        all_masks = [targets] + patch_masks
        
        # Compute weighted loss at each scale
        total_loss = 0.0
        for i, (out, mask, criterion, weight) in enumerate(
            zip(outputs, all_masks, self.criterions, self.loss_weights)
        ):
            # Ensure output matches mask size
            if out.shape[2:] != mask.shape[2:]:
                out = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(mask, out)
            total_loss += weight * loss
        
        return total_loss


# ================================================
# Simple BCE + Dice Loss (for non-deep-supervision)
# ================================================

class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss for binary segmentation.
    Simple version without asymmetric weighting.
    
    Expects input to be probabilities (after sigmoid), NOT logits.
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor):
        """
        probs:   [B, 1, H, W]  probability output (after sigmoid in model)
        targets: [B, 1, H, W]  binary mask float
        """
        # Clamp to avoid log(0)
        probs = probs.clamp(1e-7, 1 - 1e-7)
        
        bce_loss = self.bce(probs, targets)

        flat_p = probs.flatten(1)
        flat_t = targets.flatten(1)
        intersection = (flat_p * flat_t).sum(1)
        dice_loss = 1 - (2 * intersection + 1) / (flat_p.sum(1) + flat_t.sum(1) + 1)
        dice_loss = dice_loss.mean()

        return self.bce_w * bce_loss + (1 - self.bce_w) * dice_loss
