"""
Evaluation Metrics
"""
import torch
import numpy as np
from typing import Dict


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute Dice score."""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute IoU (Jaccard) score."""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
    }
