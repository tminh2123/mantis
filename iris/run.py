"""
IRIS - Training and Evaluation Script
Baseline version with MLP-only (no KAN)
"""
import os
import sys
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, load_config, parse_args
from models import IRIS, DeepSupervisionLoss, CaCLLoss, DiceBCELoss
from datasets import IRISDataset, get_dataloader
from utils import compute_metrics


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
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


def setup_logger(log_dir: str, exp_name: str, mode: str) -> logging.Logger:
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
# Stage I: Pretraining (CaCL)
# -----------------------------------------------------------------------------
def pretrain(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Stage I: Category-aware Contrastive Learning."""
    logger.info("=" * 50)
    logger.info("Stage I: Pretraining (CaCL)")
    logger.info("=" * 50)
    
    # Dataloader
    train_loader = get_dataloader(
        cfg.data.train_dir,
        batch_size=cfg.pretrain.batch_size,
        image_size=cfg.data.image_size,
        augment=True,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    
    # Loss and optimizer
    criterion = CaCLLoss(
        temperature=cfg.pretrain.temperature,
        eps=cfg.pretrain.eps,
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.weight_decay,
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(cfg.pretrain.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{cfg.pretrain.epochs}")
        
        for images, masks, texts in pbar:
            images = images.to(device)
            
            # Get contrastive embeddings
            v, l, l_raw = model.get_contrastive_embeddings(images, texts)
            
            # Compute loss
            loss = criterion(v, l, l_raw)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{cfg.pretrain.epochs} - Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, cfg.pretrain.checkpoint)
            logger.info(f"  -> Saved pretrain checkpoint (loss: {best_loss:.4f})")
    
    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")
    return model


# -----------------------------------------------------------------------------
# Stage II: Segmentation Training
# -----------------------------------------------------------------------------
def train(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Stage II: Segmentation Training with Deep Supervision."""
    logger.info("=" * 50)
    logger.info("Stage II: Segmentation Training")
    logger.info("=" * 50)
    
    # Load pretrained weights if enabled and exists
    if cfg.pretrain.enabled and os.path.exists(cfg.pretrain.checkpoint):
        ckpt = torch.load(cfg.pretrain.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded pretrained weights from {cfg.pretrain.checkpoint}")
    elif cfg.pretrain.enabled:
        logger.warning("Pretrain enabled but no checkpoint found, training from scratch")
    else:
        logger.info("Pretrain disabled, training from scratch")
    
    # Dataloaders
    train_loader = get_dataloader(
        cfg.data.train_dir,
        batch_size=cfg.train.batch_size,
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
            loss_weights=cfg.train.loss_weights,
            asymmetric_weights=cfg.train.asymmetric_weights,
            bce_weight=cfg.train.bce_weight,
        )
    else:
        criterion = DiceBCELoss(bce_weight=cfg.train.bce_weight)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    
    # Scheduler
    if cfg.train.cosine_lr:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        scheduler = None
    
    # Training loop
    best_dice = 0
    patience_counter = 0
    
    for epoch in range(cfg.train.epochs):
        # Training
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.train.epochs}")
        
        for images, masks, texts in pbar:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()  # [B, 1, H, W]
            
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
            
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, cfg.eval.threshold)
        
        logger.info(
            f"Epoch {epoch+1}/{cfg.train.epochs} - "
            f"Loss: {avg_loss:.4f}, "
            f"Val Dice: {val_metrics['dice']:.4f}, "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )
        
        # Save best
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            patience_counter = 0
            
            if epoch >= cfg.train.save_after:
                os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'metrics': val_metrics,
                }, cfg.eval.checkpoint)
                logger.info(f"  -> Saved best model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if cfg.train.early_stopping and patience_counter >= cfg.train.es_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        
        # Periodic save
        if (epoch + 1) % cfg.logging.save_every == 0:
            ckpt_path = os.path.join(cfg.logging.checkpoint_dir, f'iris_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, ckpt_path)
    
    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")
    return model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: IRIS, 
    loader: DataLoader, 
    device: torch.device, 
    threshold: float = 0.5
) -> dict:
    """Evaluate model on a dataloader."""
    model.eval()
    
    all_dice = []
    all_iou = []
    
    for images, masks, texts in loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        
        outputs = model(images, texts)
        pred = outputs['pred']
        
        # Compute metrics
        metrics = compute_metrics(pred, masks, threshold)
        all_dice.append(metrics['dice'])
        all_iou.append(metrics['iou'])
    
    return {
        'dice': np.mean(all_dice),
        'iou': np.mean(all_iou),
    }


def full_evaluate(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Full evaluation on test set with all metrics."""
    logger.info("=" * 50)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 50)
    
    # Load best weights
    if os.path.exists(cfg.eval.checkpoint):
        ckpt = torch.load(cfg.eval.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded weights from {cfg.eval.checkpoint}")
    
    # Test dataloader
    test_loader = get_dataloader(
        cfg.data.test_dir,
        batch_size=cfg.eval.batch_size,
        image_size=cfg.data.image_size,
        augment=False,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    
    # Evaluate
    metrics = evaluate(model, test_loader, device, cfg.eval.threshold)
    
    logger.info(f"Test Results:")
    logger.info(f"  Dice: {metrics['dice']*100:.2f}%")
    logger.info(f"  IoU:  {metrics['iou']*100:.2f}%")
    
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Override experiment name if provided
    if args.exp_name:
        cfg.exp_name = args.exp_name
    
    # Setup
    setup_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    
    logger = setup_logger(cfg.logging.log_dir, cfg.exp_name, args.mode)
    
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {args.mode}")
    
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
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
