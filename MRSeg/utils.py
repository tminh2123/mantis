"""
MRSeg Utilities — copied from iris/utils/ and iris/models/losses.py
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Reproducibility & device
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, strict: bool = False):
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=strict)
    else:
        model.load_state_dict(ckpt, strict=strict)
    return ckpt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred = (torch.sigmoid(pred) > threshold).float()
    flat_p = pred.flatten(1)
    flat_t = target.flatten(1)
    intersection = (flat_p * flat_t).sum(1)
    union = flat_p.sum(1) + flat_t.sum(1)
    return (2 * intersection + smooth) / (union + smooth)


def iou_score_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred = (torch.sigmoid(pred) > threshold).float()
    flat_p = pred.flatten(1)
    flat_t = target.flatten(1)
    intersection = (flat_p * flat_t).sum(1)
    union = flat_p.sum(1) + flat_t.sum(1) - intersection
    return (intersection + smooth) / (union + smooth)


def compute_metrics_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    return {
        'dice': dice_score_per_sample(pred, target, threshold),
        'iou': iou_score_per_sample(pred, target, threshold),
    }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)
