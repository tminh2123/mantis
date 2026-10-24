"""
IRIS - Training and Evaluation Script
Baseline version with MLP-only (no KAN)
"""
import os
import sys
import random
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, load_config, parse_args
from models import IRIS, DeepSupervisionLoss, CaCLLoss, DiceBCELoss
from datasets import IRISDataset, get_dataloader
from utils import compute_metrics


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: str, mode: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('iris')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(console)
    
    # File handler
    log_file = os.path.join(log_dir, f'iris_{mode}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(state: dict, filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: nn.Module):
    """Load model checkpoint."""
    ckpt = torch.load(filepath, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    return ckpt


def build_model(cfg: Config, device: torch.device) -> IRIS:
    """Build IRIS model from config."""
    model = IRIS(
        num_classes=cfg.model.num_classes,
        img_size=cfg.data.image_size,
        clip_model=cfg.model.clip_model,
        max_token_len=cfg.model.max_token_len,
        base_channel=cfg.model.base_channel,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        expansion=cfg.model.expansion,
        encoder_type=cfg.model.encoder_type,
        deep_supervision=cfg.model.deep_supervision,
        text_mask_rate=cfg.model.text_mask_rate,
        img_mask_rate=cfg.model.img_mask_rate,
    )
    return model.to(device)


# -----------------------------------------------------------------------------
# Stage I: CaCL Pretraining
# -----------------------------------------------------------------------------
def pretrain(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Stage I: Category-aware Contrastive Learning."""
    pcfg = cfg.pretrain
    
    # Dump full config to log
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"[experiment]")
    logger.info(f"  exp_name: {cfg.exp_name}")
    logger.info(f"  seed: {cfg.seed}")
    logger.info(f"[model]")
    logger.info(f"  encoder_type: {cfg.model.encoder_type}")
    logger.info(f"  base_channel: {cfg.model.base_channel}")
    logger.info(f"  expansion: {cfg.model.expansion}")
    logger.info(f"  deep_supervision: {cfg.model.deep_supervision}")
    logger.info(f"[data]")
    logger.info(f"  image_size: {cfg.data.image_size}")
    logger.info(f"  train_dir: {cfg.data.train_dir}")
    logger.info(f"[pretrain]")
    logger.info(f"  epochs: {pcfg.epochs}")
    logger.info(f"  batch_size: {pcfg.batch_size}")
    logger.info(f"  lr: {pcfg.lr}")
    logger.info(f"  temperature: {pcfg.temperature}")
    logger.info(f"  eps: {pcfg.eps}")
    logger.info("=" * 60)
    
    # Dataloader
    train_loader = get_dataloader(
        cfg.data.train_dir,
        batch_size=pcfg.batch_size,
        image_size=cfg.data.image_size,
        augment=True,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    
    # Loss
    criterion = CaCLLoss(temperature=pcfg.temperature, eps=pcfg.eps)
    
    # Optimizer - only train visual encoder + projections, language encoder is frozen
    params = [p for n, p in model.named_parameters()
              if 'lang_enc' not in n and p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    optimizer = torch.optim.Adam(params, lr=pcfg.lr, weight_decay=pcfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=pcfg.epochs)
    
    # Checkpoint dir
    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Dataset: {len(train_loader.dataset)} samples | Batch size: {pcfg.batch_size} | Steps/epoch: {len(train_loader)}")
    logger.info(f"Config: τ={pcfg.temperature}, ρ={pcfg.eps}, lr={pcfg.lr}")
    logger.info("-" * 60)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(pcfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{pcfg.epochs}", leave=False)
        for batch_idx, (images, _, texts) in enumerate(pbar):
            images = images.to(device)
            
            # Get contrastive embeddings
            v, l, l_raw = model.get_contrastive_embeddings(images, texts)
            
            # Compute loss
            loss = criterion(v, l, l_raw)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{loss_meter.avg:.4f}'})
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch [{epoch+1:02d}/{pcfg.epochs}] "
            f"Loss: {loss_meter.avg:.4f} | "
            f"LR: {lr_current:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(
                {'epoch': epoch+1, 'model': model.state_dict(), 'best_loss': best_loss},
                str(ckpt_dir / 'pretrain_best.pth')
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.logging.save_every == 0:
            save_checkpoint(
                {'epoch': epoch+1, 'model': model.state_dict()},
                str(ckpt_dir / f'pretrain_epoch{epoch+1}.pth')
            )
    
    # Save final checkpoint
    save_checkpoint(
        {'epoch': pcfg.epochs, 'model': model.state_dict(), 'best_loss': best_loss},
        pcfg.checkpoint
    )
    logger.info("-" * 60)
    logger.info(f"Pretrain done. Best loss: {best_loss:.4f}")
    logger.info(f"Saved to {pcfg.checkpoint}")
    
    return model


# -----------------------------------------------------------------------------
# Stage II: Segmentation Training
# -----------------------------------------------------------------------------
def train(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Stage II: Segmentation Training with Deep Supervision."""
    tcfg = cfg.train
    
    # Dump full config to log
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"[experiment]")
    logger.info(f"  exp_name: {cfg.exp_name}")
    logger.info(f"  seed: {cfg.seed}")
    logger.info(f"[model]")
    logger.info(f"  encoder_type: {cfg.model.encoder_type}")
    logger.info(f"  base_channel: {cfg.model.base_channel}")
    logger.info(f"  expansion: {cfg.model.expansion}")
    logger.info(f"  deep_supervision: {cfg.model.deep_supervision}")
    logger.info(f"[data]")
    logger.info(f"  image_size: {cfg.data.image_size}")
    logger.info(f"  train_dir: {cfg.data.train_dir}")
    logger.info(f"  val_dir: {cfg.data.val_dir}")
    logger.info(f"[train]")
    logger.info(f"  epochs: {tcfg.epochs}")
    logger.info(f"  batch_size: {tcfg.batch_size}")
    logger.info(f"  lr: {tcfg.lr}")
    logger.info(f"  early_stopping: {tcfg.early_stopping}")
    logger.info(f"  es_patience: {tcfg.es_patience}")
    logger.info("=" * 60)
    
    # Load pretrained weights if available
    if cfg.pretrain.enabled and Path(cfg.pretrain.checkpoint).exists():
        load_checkpoint(cfg.pretrain.checkpoint, model)
        logger.info(f"Loaded pretrained weights from {cfg.pretrain.checkpoint}")
    elif cfg.pretrain.enabled:
        logger.warning("Pretrain enabled but no checkpoint found, training from scratch")
    else:
        logger.info("Pretrain disabled, training from scratch")
    
    # Dataloaders
    train_loader = get_dataloader(
        cfg.data.train_dir,
        batch_size=tcfg.batch_size,
        image_size=cfg.data.image_size,
        augment=True,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    
    val_loader = get_dataloader(
        cfg.data.val_dir,
        batch_size=cfg.eval.batch_size,
        image_size=cfg.data.image_size,
        augment=False,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    
    # Loss
    if cfg.model.deep_supervision:
        criterion = DeepSupervisionLoss(
            loss_weights=tcfg.loss_weights,
            asymmetric_weights=tcfg.asymmetric_weights,
            bce_weight=tcfg.bce_weight,
        )
        logger.info(f"Using Deep Supervision Loss with weights: {tcfg.loss_weights}")
    else:
        criterion = DiceBCELoss(bce_weight=tcfg.bce_weight)
        logger.info("Using DiceBCE Loss")
    
    # Optimizer with different LR for visual encoder (like c2seg)
    visual_encoder_prefixes = ('down0', 'down1', 'down2', 'down3', 'down4', 'pool')
    visual_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('lang_enc'):
            continue  # frozen
        if any(n.startswith(prefix) for prefix in visual_encoder_prefixes):
            visual_params.append(p)
        else:
            other_params.append(p)
    
    optimizer = torch.optim.Adam([
        {'params': visual_params, 'lr': tcfg.lr * 0.1},  # small lr for visual encoder
        {'params': other_params,  'lr': tcfg.lr},
    ], weight_decay=tcfg.weight_decay)
    
    # Scheduler
    if tcfg.cosine_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
        logger.info("Using CosineAnnealingWarmRestarts scheduler (T_0=10, eta_min=1e-4)")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    
    # Checkpoint dir
    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")
    logger.info(f"Batch size: {tcfg.batch_size} | Steps/epoch: {len(train_loader)}")
    logger.info("-" * 60)
    
    # Training loop
    best_dice = 0.0
    no_improve = 0
    
    for epoch in range(tcfg.epochs):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg.epochs}", leave=False)
        for batch_idx, (images, masks, texts) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)  # [B, 1, H, W]
            
            # Forward
            outputs = model(images, texts)
            
            # Compute loss
            if cfg.model.deep_supervision and 'aux_preds' in outputs:
                loss = criterion(outputs['pred'], outputs['aux_preds'], masks)
            else:
                loss = criterion(outputs['pred'], masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            if tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{loss_meter.avg:.4f}'})
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Validation
        metrics = evaluate(model, val_loader, device, cfg.eval.threshold)
        
        logger.info(
            f"Epoch [{epoch+1}/{tcfg.epochs}] Loss: {loss_meter.avg:.4f} | "
            f"Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            no_improve = 0
            save_checkpoint(
                {'epoch': epoch+1, 'model': model.state_dict(), 'best_dice': best_dice},
                str(ckpt_dir / 'best.pth')
            )
        else:
            no_improve += 1
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.logging.save_every == 0:
            save_checkpoint(
                {'epoch': epoch+1, 'model': model.state_dict()},
                str(ckpt_dir / f'epoch{epoch+1}.pth')
            )
        
        # Early stopping
        if tcfg.early_stopping and no_improve >= tcfg.es_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {no_improve} epochs)")
            break
    
    logger.info("-" * 60)
    logger.info(f"Training done. Best Dice: {best_dice:.2f}%")
    
    return model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: IRIS, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    """Evaluate model on a dataloader."""
    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    for images, masks, texts in loader:
        images = images.to(device)
        masks = masks.to(device).float()
        
        outputs = model(images, texts)
        pred = outputs['pred']
        
        # Compute metrics
        metrics = compute_metrics(pred, masks.unsqueeze(1), threshold)
        B = images.size(0)
        dice_meter.update(metrics['dice'] * 100, B)  # Convert to percentage
        iou_meter.update(metrics['iou'] * 100, B)
    
    return {
        'dice': dice_meter.avg,
        'iou': iou_meter.avg,
    }


def full_evaluate(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Full evaluation on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    # Load best weights
    ckpt_path = cfg.eval.checkpoint
    if Path(ckpt_path).exists():
        ckpt = load_checkpoint(ckpt_path, model)
        logger.info(f"Loaded weights from {ckpt_path}")
        if 'epoch' in ckpt:
            logger.info(f"  Checkpoint from epoch: {ckpt.get('epoch', 'N/A')}")
        if 'best_dice' in ckpt:
            logger.info(f"  Val Dice at save: {ckpt.get('best_dice', 'N/A'):.2f}%")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}")
    
    # Test dataloader
    test_loader = get_dataloader(
        cfg.data.test_dir,
        batch_size=cfg.eval.batch_size,
        image_size=cfg.data.image_size,
        augment=False,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    metrics = evaluate(model, test_loader, device, cfg.eval.threshold)
    
    logger.info("-" * 60)
    logger.info(f"Test Results | Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}%")
    logger.info("-" * 60)
    
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args)
    
    # Setup
    setup_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    
    logger = setup_logger(cfg.logging.log_dir, args.mode)
    
    # Log start
    logger.info("=" * 60)
    logger.info(f"IRIS - {args.mode.upper()}")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Build model
    model = build_model(cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Run
    if args.mode == 'pretrain':
        if cfg.pretrain.enabled:
            pretrain(cfg, model, device, logger)
        else:
            logger.info("Pretrain is disabled in config, skipping")
        
    elif args.mode == 'train':
        train(cfg, model, device, logger)
        
    elif args.mode == 'eval':
        full_evaluate(cfg, model, device, logger)
        
    elif args.mode == 'full':
        if cfg.pretrain.enabled:
            pretrain(cfg, model, device, logger)
        else:
            logger.info("Pretrain is disabled, skipping Stage I")
        train(cfg, model, device, logger)
        full_evaluate(cfg, model, device, logger)
    
    logger.info("=" * 60)
    logger.info(f"Done! End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
