"""
Segmentation metrics: Dice, mIoU, HD95, ASSD
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """pred/target: [B, 1, H, W] float tensors (probs after sigmoid)."""
    # Model output already has sigmoid, so just threshold directly
    pred = (pred > threshold).float()
    flat_p = pred.flatten(1)
    flat_t = target.flatten(1)
    intersection = (flat_p * flat_t).sum(1)
    score = (2 * intersection + 1) / (flat_p.sum(1) + flat_t.sum(1) + 1)
    return score.mean().item() * 100


def miou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """pred/target: [B, 1, H, W] float tensors (probs after sigmoid)."""
    pred = (pred > threshold).float()
    flat_p = pred.flatten(1)
    flat_t = target.flatten(1)
    intersection = (flat_p * flat_t).sum(1)
    union = flat_p.sum(1) + flat_t.sum(1) - intersection
    iou = (intersection + 1) / (union + 1)
    return iou.mean().item() * 100


def _erode(arr: np.ndarray) -> np.ndarray:
    """Binary erosion with edge case handling."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(arr)
    # If erosion removes everything, return original (for very thin structures)
    if eroded is None or not eroded.any():
        return np.zeros_like(arr, dtype=bool)
    return eroded


def _surface_distances(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Compute surface distances between two binary masks."""
    # Ensure 2D arrays
    if pred_bin.ndim != 2 or gt_bin.ndim != 2:
        return np.array([]), np.array([])
    
    # Get borders via erosion XOR
    pred_eroded = _erode(pred_bin)
    gt_eroded = _erode(gt_bin)
    
    pred_border = pred_bin ^ pred_eroded
    gt_border = gt_bin ^ gt_eroded
    
    # If border extraction failed, use contour-based approach
    if not pred_border.any():
        # Fall back: use edge pixels (where mask meets background)
        from scipy.ndimage import binary_dilation
        pred_dilated = binary_dilation(pred_bin)
        pred_border = pred_dilated & ~pred_bin
        if not pred_border.any():
            pred_border = pred_bin  # Use entire mask as fallback
    
    if not gt_border.any():
        from scipy.ndimage import binary_dilation
        gt_dilated = binary_dilation(gt_bin)
        gt_border = gt_dilated & ~gt_bin
        if not gt_border.any():
            gt_border = gt_bin

    # Distance transforms
    dt_pred = distance_transform_edt(~pred_bin)
    dt_gt = distance_transform_edt(~gt_bin)

    # Surface distances
    sds_pred2gt = dt_gt[pred_border]
    sds_gt2pred = dt_pred[gt_border]
    
    return sds_pred2gt, sds_gt2pred


def hd95(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Hausdorff Distance 95th percentile.
    pred/target: [B, 1, H, W] float tensors (probs after sigmoid).
    """
    pred_bin = (pred > threshold).cpu().numpy().astype(bool)
    gt_bin = (target > threshold).cpu().numpy().astype(bool)

    scores = []
    for p, g in zip(pred_bin, gt_bin):
        # Handle channel dimension
        if p.ndim == 3:
            p = p[0]
        if g.ndim == 3:
            g = g[0]
        
        # Skip if either mask is empty
        if p.sum() == 0 or g.sum() == 0:
            continue
        
        d1, d2 = _surface_distances(p, g)
        if len(d1) == 0 and len(d2) == 0:
            continue
        
        # Concatenate available distances
        if len(d1) > 0 and len(d2) > 0:
            all_d = np.concatenate([d1, d2])
        elif len(d1) > 0:
            all_d = d1
        else:
            all_d = d2
            
        scores.append(np.percentile(all_d, 95))
    
    return float(np.mean(scores)) if scores else 0.0


def assd(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Average Symmetric Surface Distance.
    pred/target: [B, 1, H, W] float tensors (probs after sigmoid).
    """
    pred_bin = (pred > threshold).cpu().numpy().astype(bool)
    gt_bin = (target > threshold).cpu().numpy().astype(bool)

    scores = []
    for p, g in zip(pred_bin, gt_bin):
        # Handle channel dimension
        if p.ndim == 3:
            p = p[0]
        if g.ndim == 3:
            g = g[0]
        
        # Skip if either mask is empty
        if p.sum() == 0 or g.sum() == 0:
            continue
        
        d1, d2 = _surface_distances(p, g)
        if len(d1) == 0 and len(d2) == 0:
            continue
        
        # Compute ASSD from available distances
        if len(d1) > 0 and len(d2) > 0:
            assd_val = (d1.mean() + d2.mean()) / 2
        elif len(d1) > 0:
            assd_val = d1.mean()
        else:
            assd_val = d2.mean()
            
        scores.append(assd_val)
    
    return float(np.mean(scores)) if scores else 0.0
