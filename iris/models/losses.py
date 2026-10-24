"""
Loss Functions for IRIS
- CaCL Loss (Stage I - Pretraining)
- Deep Supervision Loss (Stage II - Training)
- Dice + BCE Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
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
    """Combined Dice + BCE loss."""
    
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class PatchMaskGenerator(nn.Module):
    """Generate coarse ground truth masks for deep supervision."""
    
    def __init__(self, patch_sizes: List[int] = [2, 4, 8, 16]):
        super().__init__()
        self.patch_sizes = patch_sizes
    
    def forward(self, mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale GT masks.
        
        Args:
            mask: Full resolution mask [B, 1, H, W]
        
        Returns:
            List of coarse masks at different scales
        """
        coarse_masks = []
        
        for patch_size in self.patch_sizes:
            # Average pooling to get patch-level mask
            pooled = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
            # Binarize: if any pixel in patch is positive, patch is positive
            coarse = (pooled > 0).float()
            coarse_masks.append(coarse)
        
        return coarse_masks


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss with asymmetric weights.
    
    Computes loss at multiple scales with decreasing weights
    for coarser predictions.
    """
    
    def __init__(
        self,
        loss_weights: List[float] = [0.5, 0.2, 0.15, 0.1, 0.05],
        asymmetric_weights: List[Tuple[float, float]] = None,
        bce_weight: float = 0.5,
    ):
        super().__init__()
        
        self.loss_weights = loss_weights
        self.asymmetric_weights = asymmetric_weights or [
            (0.5, 0.5),   # Scale 0: balanced
            (0.4, 0.6),   # Scale 1: slight positive bias
            (0.3, 0.7),   # Scale 2: moderate positive bias
            (0.2, 0.8),   # Scale 3: strong positive bias
            (0.1, 0.9),   # Scale 4: very strong positive bias
        ]
        self.bce_weight = bce_weight
        
        self.dice = DiceLoss()
        self.patch_generator = PatchMaskGenerator([2, 4, 8, 16])
    
    def _weighted_bce(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        neg_weight: float, 
        pos_weight: float
    ) -> torch.Tensor:
        """BCE with asymmetric class weights."""
        weights = torch.where(target > 0.5, pos_weight, neg_weight)
        bce = F.binary_cross_entropy_with_logits(pred, target, weight=weights)
        return bce
    
    def forward(
        self,
        pred: torch.Tensor,
        aux_preds: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute deep supervision loss.
        
        Args:
            pred: Main prediction [B, C, H, W]
            aux_preds: List of auxiliary predictions [f1, f2, f3, f4]
            target: Ground truth mask [B, C, H, W]
        
        Returns:
            Weighted sum of losses
        """
        # Main loss at full resolution
        neg_w, pos_w = self.asymmetric_weights[0]
        main_loss = self.dice(pred, target) + self.bce_weight * self._weighted_bce(pred, target, neg_w, pos_w)
        total_loss = self.loss_weights[0] * main_loss
        
        # Generate coarse GT masks
        coarse_targets = self.patch_generator(target)
        
        # Auxiliary losses
        for i, aux_pred in enumerate(aux_preds):
            coarse_target = coarse_targets[i]
            
            # Resize prediction to match coarse target
            if aux_pred.shape[2:] != coarse_target.shape[2:]:
                aux_pred = F.interpolate(
                    aux_pred, 
                    size=coarse_target.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            neg_w, pos_w = self.asymmetric_weights[i + 1]
            aux_loss = self.dice(aux_pred, coarse_target) + self.bce_weight * self._weighted_bce(aux_pred, coarse_target, neg_w, pos_w)
            total_loss = total_loss + self.loss_weights[i + 1] * aux_loss
        
        return total_loss


class CaCLLoss(nn.Module):
    """
    Category-aware Contrastive Learning Loss (Stage I).
    
    Aligns image and text embeddings with temperature scaling
    and epsilon smoothing.
    """
    
    def __init__(self, temperature: float = 0.07, eps: float = 0.8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(
        self, 
        v: torch.Tensor, 
        l: torch.Tensor, 
        l_raw: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CaCL loss.
        
        Args:
            v: Image embeddings [B, D] (normalized)
            l: Text embeddings [B, D] (normalized, from learned projection)
            l_raw: Raw text embeddings [B, D] (normalized, from CLIP)
        
        Returns:
            Contrastive loss
        """
        B = v.size(0)
        device = v.device
        
        # Similarity matrices
        sim_v2l = v @ l.T / self.temperature      # [B, B]
        sim_l2v = l @ v.T / self.temperature      # [B, B]
        
        # Soft labels from raw CLIP text similarities
        raw_sim = l_raw @ l_raw.T                 # [B, B]
        soft_labels = F.softmax(raw_sim / self.temperature, dim=-1)
        
        # Epsilon smoothing
        hard_labels = torch.eye(B, device=device)
        labels = self.eps * soft_labels + (1 - self.eps) * hard_labels
        
        # Cross entropy losses
        loss_v2l = -torch.sum(labels * F.log_softmax(sim_v2l, dim=-1), dim=-1).mean()
        loss_l2v = -torch.sum(labels * F.log_softmax(sim_l2v, dim=-1), dim=-1).mean()
        
        return (loss_v2l + loss_l2v) / 2


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """Get loss function by name."""
    if name == "deep_supervision":
        return DeepSupervisionLoss(**kwargs)
    elif name == "dice_bce":
        return DiceBCELoss(**kwargs)
    elif name == "cacl":
        return CaCLLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {name}")
