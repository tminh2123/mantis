# -*- coding: utf-8 -*-
"""
Loss functions and evaluation metrics for medical image segmentation tasks.
Learning rate schedulers and utility functions for text/image preprocessing.
"""

import math
import warnings
import weakref
from functools import wraps

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from numpy import average, dot, linalg
from sklearn.metrics import roc_auc_score, jaccard_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


# ================================================
# Loss Functions
# ================================================
class WeightedBCE(nn.Module):
    """Weighted Binary Cross-Entropy Loss."""
    def __init__(self, weights=(0.5, 0.5)):
        super().__init__()
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        assert logits.shape == targets.shape

        loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pos = (targets > 0.5).float()
        neg = 1.0 - pos

        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12

        weighted_loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

        return weighted_loss


class WeightedDiceLoss(nn.Module):
    """Weighted Dice Loss."""
    def __init__(self, weights=(0.5, 0.5)):
        super().__init__()
        self.weights = weights

    def forward(self, targets: torch.Tensor, logits: torch.Tensor, smooth=1e-5) -> torch.Tensor:
        b = logits.size(0)
        targets = targets.view(b, -1)
        logits = logits.view(b, -1)
        assert targets.shape == logits.shape

        weights = targets.detach() * (self.weights[0] - self.weights[1]) + self.weights[1]
        targets, logits = weights * targets, weights * logits

        intersection = (targets * logits).sum(-1)
        union = (targets * targets).sum(-1) + (logits * logits).sum(-1)
        dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

        return dice_loss.mean()


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss."""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = targets.size(0)
        smooth = 1.0
        logits_flat = logits.view(n, -1)
        targets_flat = targets.view(n, -1)

        intersection = logits_flat * targets_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (logits_flat.sum(1) + targets_flat.sum(1) + smooth)

        return 1 - dice_score.mean()


class MultiClassDiceLoss(nn.Module):
    """Multi-class Dice Loss (fixed number of classes = 5)."""
    def __init__(self):
        super().__init__()
        self.dice_loss = WeightedDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape, "Logits & targets shape mismatch"
        total_loss = 0.0
        for i in range(5):  # hard-coded 5 classes
            total_loss += self.dice_loss(logits[:, i], targets[:, i])
        return total_loss / 5


class DiceLoss(nn.Module):
    """General Dice Loss with optional class weighting."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Convert class indices to one-hot encoding."""
        return torch.cat([(input_tensor == i).unsqueeze(1) for i in range(self.n_classes)], dim=1).float()

    @staticmethod
    def _dice_loss(score: torch.Tensor, target: torch.Tensor, smooth=1e-5) -> torch.Tensor:
        intersect = torch.sum(score * target)
        denom = torch.sum(score * score) + torch.sum(target * target)
        return 1 - (2 * intersect + smooth) / (denom + smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weight=None, softmax=False):
        if softmax:
            logits = torch.softmax(logits, dim=1)
        targets = self._one_hot_encoder(targets)

        if weight is None:
            weight = [1.0] * self.n_classes

        assert logits.size() == targets.size(), "Shape mismatch"
        loss, dice_scores = 0.0, []

        for i in range(self.n_classes):
            dice = self._dice_loss(logits[:, i], targets[:, i])
            dice_scores.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes, *dice_scores[1:4]  # return first 3 class dice


class WeightedDiceCE(nn.Module):
    """
    Combined Weighted Dice + Cross Entropy Loss.
    Typically used for multi-class segmentation.
    """
    def __init__(self, n_classes=4, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def _show_dice(self, logits: torch.Tensor, targets: torch.Tensor):
        dice, dice1, dice2, dice3 = self.dice_loss(logits, targets)
        return 1 - dice, 1 - dice1, 1 - dice2, 1 - dice3

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        dice_val = self.dice_loss(logits, targets)
        ce_val = self.ce_loss(logits, targets)
        return self.dice_weight * dice_val + self.ce_weight * ce_val


class WeightedDiceBCE_unsup(nn.Module):
    """
    Unsupervised Dice + BCE Loss with additional LV regularization.
    """
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_loss = WeightedDiceLoss((0.5, 0.5))
        self.bce_loss = WeightedBCE((0.5, 0.5))
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def _show_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = (logits >= 0.5).float()
        targets = (targets > 0).float()
        return 1.0 - self.dice_loss(preds, targets)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, lv_loss: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce + 0.1 * lv_loss


class WeightedDiceBCE(nn.Module):
    """
    Combined Weighted Dice + Binary Cross-Entropy Loss.
    Typically used for binary segmentation.
    """
    def __init__(self, weight=(0.5, 0.5)):
        super().__init__()
        self.dice_loss = WeightedDiceLoss(weight)
        self.bce_loss = WeightedBCE(weight)

    def _show_dice(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        targets = (targets > 0).float()
        preds = (logits >= 0.5).float()
        return 1.0 - self.dice_loss(targets, preds)

    def forward(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(targets, logits)
        bce = self.bce_loss(logits, targets)
        return 0.5 * dice + 0.5 * bce


# ================================================
# Evaluation Metrics
# ================================================
def auc_on_batch(masks: torch.Tensor, preds: torch.Tensor) -> float:
    """Compute mean ROC-AUC score over a batch."""
    aucs = []
    for i in range(preds.size(0)):
        aucs.append(
            roc_auc_score(
                masks[i].cpu().numpy().ravel(),
                preds[i][0].detach().cpu().numpy().ravel()
            )
        )
    return float(np.mean(aucs))


def r2_on_batch(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Compute R^2 metric (scaled) over a batch."""
    r2_scores = []
    for i in range(preds.size(0)):
        pred, mask = preds[i][0].view(-1), masks[i][0].view(-1)
        mask_mean = mask.mean()
        ss_total = ((mask - mask_mean) ** 2).sum()
        ss_res = ((pred - mask) ** 2).sum()
        r2_scores.append(torch.exp(-ss_res / (ss_total + 1e-5)))
    return torch.stack(r2_scores).mean()


def iou_on_batch(masks: torch.Tensor, preds: torch.Tensor) -> float:
    """Compute mean IoU (Jaccard index) over a batch."""
    ious = []
    for i in range(preds.size(0)):
        mask = (masks[i].cpu().detach().numpy() > 0).astype(int)
        pred = (preds[i][0].cpu().detach().numpy() >= 0.5).astype(int)
        ious.append(jaccard_score(mask.ravel(), pred.ravel(), zero_division=1.0))
    return float(np.mean(ious))


def dice_coef(y_true: np.ndarray, y_pred: np.ndarray, smooth=1e-5) -> float:
    """Compute Dice coefficient."""
    y_true_f, y_pred_f = y_true.ravel(), y_pred.ravel()
    intersect = np.sum(y_true_f * y_pred_f)
    return (2. * intersect + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_on_batch(masks: torch.Tensor, preds: torch.Tensor) -> float:
    """Compute mean Dice coefficient over a batch."""
    dices = []
    for i in range(preds.size(0)):
        mask = (masks[i].cpu().detach().numpy() > 0).astype(int)
        pred = (preds[i][0].cpu().detach().numpy() >= 0.5).astype(int)
        dices.append(dice_coef(mask, pred))
    return float(np.mean(dices))


# ================================================
# Custom Learning Rate Schedulers
# ================================================
class _LRScheduler:
    """Base class for learning rate schedulers."""
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # Save initial learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(f"'initial_lr' not specified in param_groups[{i}]")

        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]

        # Ensure optimizer.step is wrapped with counter
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                return method
            instance_ref = weakref.ref(method.__self__)
            func, cls = method.__func__, instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                return func.__get__(instance, cls)(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self) -> dict:
        """Return scheduler state dict (excluding optimizer)."""
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rates."""
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch: int = None):
        """Update learning rates at each step."""
        if self._step_count == 1 and not hasattr(self.optimizer.step, "_with_counter"):
            warnings.warn(
                "Detected `lr_scheduler.step()` before `optimizer.step()`. "
                "Call them in the order: optimizer.step() before lr_scheduler.step().",
                UserWarning,
            )

        self._step_count += 1
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts (SGDR).
    https://arxiv.org/abs/1608.03983
    """
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0, last_epoch: int = -1):
        if T_0 <= 0:
            raise ValueError(f"Expected positive integer T_0, got {T_0}")
        if T_mult < 1:
            raise ValueError(f"Expected T_mult >= 1, got {T_mult}")

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min +
            (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: float = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log(epoch / self.T_0 * (self.T_mult - 1) + 1, self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_cur, self.T_i = epoch, self.T_0

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


# ================================================
# Text Processing
# ================================================
def read_text(filename: str, pad_length: int = 9) -> dict:
    """
    Read descriptions from Excel file and pad short texts.
    Returns dict: {image_name: description}.
    """
    df = pd.read_excel(filename, engine="openpyxl")
    text_dict = {}
    for _, row in df.iterrows():
        desc = str(row["Description"])
        count = len(desc.split())
        if count < pad_length:
            desc += " EOF XXX" * (pad_length - count)
        text_dict[row["Image"]] = desc
    return text_dict


def read_text_lv(filename: str, pad_length: int = 30) -> dict:
    """
    Read descriptions for LV loss and pad to fixed length.
    Returns dict: {image_name: description}.
    """
    df = pd.read_excel(filename)
    text_dict = {}
    for _, row in df.iterrows():
        desc = str(row["Description"])
        count = len(desc.split())
        if count < pad_length:
            desc += " EOF XXX" * (pad_length - count)
        text_dict[row["Image"]] = desc
    return text_dict


# ================================================
# Image Processing
# ================================================
def get_thumb(image: Image.Image, size=(224, 224), greyscale=False) -> Image.Image:
    """Resize image and optionally convert to grayscale."""
    image = image.resize(size, Image.Resampling.LANCZOS)
    if greyscale:
        image = image.convert("L")
    return image


def img_similarity(image1: Image.Image, image2: Image.Image) -> float:
    """Compute cosine similarity between two images."""
    img1, img2 = get_thumb(image1), get_thumb(image2)
    vectors, norms = [], []

    for img in (img1, img2):
        vec = [average(pixel) for pixel in img.getdata()]
        vectors.append(vec)
        norms.append(linalg.norm(vec, 2))

    a, b = vectors
    a_norm, b_norm = norms
    return float(dot(a / a_norm, b / b_norm))


def save_on_batch(images, masks, preds, names, vis_path: str):
    """Save prediction and ground truth masks as images."""
    for i in range(preds.shape[0]):
        pred = (preds[i][0].cpu().detach().numpy() >= 0.5).astype(np.uint8) * 255
        mask = (masks[i].cpu().detach().numpy() > 0).astype(np.uint8) * 255
        cv2.imwrite(f"{vis_path}{names[i][:-4]}_pred.jpg", pred)
        cv2.imwrite(f"{vis_path}{names[i][:-4]}_gt.jpg", mask)

