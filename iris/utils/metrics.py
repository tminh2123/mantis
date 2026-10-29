"""
Evaluation Metrics
"""
import torch
import numpy as np
from typing import Dict, Union


def dice_score_per_sample(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice score per sample.
    pred: [B, 1, H, W] raw logits
    target: [B, 1, H, W] binary mask
    Returns: [B] tensor of per-sample Dice scores
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dims, keep batch dim
    flat_p = pred.flatten(1)  # [B, H*W]
    flat_t = target.flatten(1)  # [B, H*W]
    
    intersection = (flat_p * flat_t).sum(1)  # [B]
    union = flat_p.sum(1) + flat_t.sum(1)  # [B]
    
    dice = (2 * intersection + smooth) / (union + smooth)  # [B]
    
    return dice


def iou_score_per_sample(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU (Jaccard) score per sample.
    pred: [B, 1, H, W] raw logits
    target: [B, 1, H, W] binary mask
    Returns: [B] tensor of per-sample IoU scores
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dims, keep batch dim
    flat_p = pred.flatten(1)  # [B, H*W]
    flat_t = target.flatten(1)  # [B, H*W]
    
    intersection = (flat_p * flat_t).sum(1)  # [B]
    union = flat_p.sum(1) + flat_t.sum(1) - intersection  # [B]
    
    iou = (intersection + smooth) / (union + smooth)  # [B]
    
    return iou


def compute_metrics_per_sample(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Compute per-sample metrics for proper Macro averaging.
    Returns dict with 'dice' and 'iou' tensors of shape [B].
    """
    return {
        'dice': dice_score_per_sample(pred, target, threshold),
        'iou': iou_score_per_sample(pred, target, threshold),
    }


# Legacy functions for backward compatibility
def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute Dice score (batch mean). Use dice_score_per_sample for proper Macro."""
    return dice_score_per_sample(pred, target, threshold, smooth).mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute IoU score (batch mean). Use iou_score_per_sample for proper Macro."""
    return iou_score_per_sample(pred, target, threshold, smooth).mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all metrics (batch mean). Use compute_metrics_per_sample for proper Macro."""
    return {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
    }
