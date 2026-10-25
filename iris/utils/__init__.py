import random
import numpy as np
import torch
import yaml
from pathlib import Path

from .metrics import compute_metrics, dice_score, iou_score


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    """Load YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(state: dict, path: str):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, strict: bool = False):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=strict)
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=strict)
    else:
        model.load_state_dict(ckpt, strict=strict)
    return ckpt


__all__ = [
    'compute_metrics', 'dice_score', 'iou_score',
    'set_seed', 'load_config', 'get_device', 'save_checkpoint', 'load_checkpoint',
]
