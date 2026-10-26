"""
Evaluation Metrics

Uses MACRO averaging (per-sample then mean) to match c2seg-origin paper.
"""
import torch
import numpy as np
from typing import Dict


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Compute Dice score - MACRO (per-sample average).
    
    This matches c2seg-origin's dice_on_batch() which computes dice per sample
    then averages, rather than computing over all pixels in batch.
    
    Args:
        pred: [B, 1, H, W] logits (before sigmoid)
        target: [B, 1, H, W] binary mask
        threshold: threshold for binarization
        smooth: smoothing factor to avoid division by zero
    
    Returns:
        Mean dice score across samples in batch
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dims but keep batch dim: [B, H*W]
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    
    # Per-sample intersection and union
    intersection = (pred_flat * target_flat).sum(1)  # [B]
    union = pred_flat.sum(1) + target_flat.sum(1)    # [B]
    
    # Per-sample dice
    dice = (2 * intersection + smooth) / (union + smooth)  # [B]
    
    return dice.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Compute IoU (Jaccard) score - MACRO (per-sample average).
    
    Args:
        pred: [B, 1, H, W] logits (before sigmoid)
        target: [B, 1, H, W] binary mask
        threshold: threshold for binarization
        smooth: smoothing factor to avoid division by zero
    
    Returns:
        Mean IoU score across samples in batch
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dims but keep batch dim: [B, H*W]
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    
    # Per-sample intersection and union
    intersection = (pred_flat * target_flat).sum(1)  # [B]
    union = pred_flat.sum(1) + target_flat.sum(1) - intersection  # [B]
    
    # Per-sample IoU
    iou = (intersection + smooth) / (union + smooth)  # [B]
    
    return iou.mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all metrics (MACRO averaging)."""
    return {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
    }
