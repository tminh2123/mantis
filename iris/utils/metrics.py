"""
Evaluation Metrics
"""
import torch
import numpy as np
from typing import Dict


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute Dice score (Macro: per-sample then mean).
    pred: [B, 1, H, W] raw logits
    target: [B, 1, H, W] binary mask
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dims, keep batch dim
    flat_p = pred.flatten(1)  # [B, H*W]
    flat_t = target.flatten(1)  # [B, H*W]
    
    intersection = (flat_p * flat_t).sum(1)  # [B]
    union = flat_p.sum(1) + flat_t.sum(1)  # [B]
    
    dice = (2 * intersection + smooth) / (union + smooth)  # [B]
    
    return dice.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute IoU (Jaccard) score (Macro: per-sample then mean).
    pred: [B, 1, H, W] raw logits
    target: [B, 1, H, W] binary mask
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dims, keep batch dim
    flat_p = pred.flatten(1)  # [B, H*W]
    flat_t = target.flatten(1)  # [B, H*W]
    
    intersection = (flat_p * flat_t).sum(1)  # [B]
    union = flat_p.sum(1) + flat_t.sum(1) - intersection  # [B]
    
    iou = (intersection + smooth) / (union + smooth)  # [B]
    
    return iou.mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all metrics (Macro averaging)."""
    return {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
    }
