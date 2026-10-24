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
# GPU Memory Utilities
# -----------------------------------------------------------------------------
def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
        'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
    }


def reset_gpu_memory_stats():
    """Reset GPU memory stats for tracking max memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


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


def log_config(cfg: Config, logger: logging.Logger):
    """Log configuration summary."""
    logger.info("=" * 60)
    logger.info("Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info("-" * 40)
    logger.info("Model:")
    logger.info(f"  encoder_type: {cfg.model.encoder_type}")
    logger.info(f"  base_channel: {cfg.model.base_channel}")
    logger.info(f"  num_heads: {cfg.model.num_heads}")
    logger.info(f"  expansion: {cfg.model.expansion}")
    logger.info(f"  deep_supervision: {cfg.model.deep_supervision}")
    logger.info(f"  clip_model: {cfg.model.clip_model}")
    logger.info("-" * 40)
    logger.info("Data:")
    logger.info(f"  image_size: {cfg.data.image_size}")
    logger.info(f"  train_dir: {cfg.data.train_dir}")
    logger.info(f"  val_dir: {cfg.data.val_dir}")
    logger.info(f"  test_dir: {cfg.data.test_dir}")
    logger.info("-" * 40)
    logger.info("Pretrain:")
    logger.info(f"  enabled: {cfg.pretrain.enabled}")
    logger.info(f"  epochs: {cfg.pretrain.epochs}")
    logger.info(f"  batch_size: {cfg.pretrain.batch_size}")
    logger.info(f"  lr: {cfg.pretrain.lr}")
    logger.info("-" * 40)
    logger.info("Train:")
    logger.info(f"  epochs: {cfg.train.epochs}")
    logger.info(f"  batch_size: {cfg.train.batch_size}")
    logger.info(f"  lr: {cfg.train.lr}")
    logger.info(f"  early_stopping: {cfg.train.early_stopping}")
    logger.info(f"  es_patience: {cfg.train.es_patience}")
    logger.info("-" * 40)
    logger.info("Output:")
    logger.info(f"  log_dir: {cfg.logging.log_dir}")
    logger.info(f"  checkpoint_dir: {cfg.logging.checkpoint_dir}")
    logger.info("=" * 60)


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
    logger.info("=" * 60)
    logger.info("Stage I: Pretraining (CaCL)")
    logger.info("=" * 60)
    
    # Reset GPU memory stats
    reset_gpu_memory_stats()
    stage_start_time = time.time()
    
    # Dataloader
    train_loader = get_dataloader(
        cfg.data.train_dir,
        batch_size=cfg.pretrain.batch_size,
        image_size=cfg.data.image_size,
        augment=True,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info(f"Batch size: {cfg.pretrain.batch_size}")
    
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
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{cfg.pretrain.epochs}")
        
        for batch_idx, (images, masks, texts) in enumerate(pbar):
            images = images.to(device)
            
            # Get contrastive embeddings
            v, l, l_raw = model.get_contrastive_embeddings(images, texts)
            
            # Compute loss
            loss = criterion(v, l, l_raw)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar with memory info
            mem = get_gpu_memory_info()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mem': f'{mem["allocated"]:.1f}GB',
            })
        
        scheduler.step()
        
        # Epoch stats
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        mem = get_gpu_memory_info()
        
        logger.info(
            f"Epoch {epoch+1}/{cfg.pretrain.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {format_time(epoch_time)} | "
            f"GPU: {mem['allocated']:.1f}GB | "
            f"MaxGPU: {mem['max_allocated']:.1f}GB"
        )
        
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
    
    # Stage summary
    stage_time = time.time() - stage_start_time
    mem = get_gpu_memory_info()
    logger.info("-" * 40)
    logger.info(f"Pretraining Complete")
    logger.info(f"  Best Loss: {best_loss:.4f}")
    logger.info(f"  Total Time: {format_time(stage_time)}")
    logger.info(f"  Max GPU Memory: {mem['max_allocated']:.2f} GB")
    logger.info("-" * 40)
    
    return model


# -----------------------------------------------------------------------------
# Stage II: Segmentation Training
# -----------------------------------------------------------------------------
def train(cfg: Config, model: IRIS, device: torch.device, logger: logging.Logger):
    """Stage II: Segmentation Training with Deep Supervision."""
    logger.info("=" * 60)
    logger.info("Stage II: Segmentation Training")
    logger.info("=" * 60)
    
    # Reset GPU memory stats
    reset_gpu_memory_stats()
    stage_start_time = time.time()
    
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
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info(f"Batch size: {cfg.train.batch_size}")
    
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
        epoch_start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{cfg.train.epochs}")
        
        for batch_idx, (images, masks, texts) in enumerate(pbar):
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
            num_batches += 1
            
            # Update progress bar with memory info
            mem = get_gpu_memory_info()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mem': f'{mem["allocated"]:.1f}GB',
            })
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / num_batches
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, cfg.eval.threshold)
        
        # Epoch stats
        epoch_time = time.time() - epoch_start_time
        mem = get_gpu_memory_info()
        lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch+1}/{cfg.train.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {format_time(epoch_time)} | "
            f"GPU: {mem['allocated']:.1f}GB | "
            f"MaxGPU: {mem['max_allocated']:.1f}GB"
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
            logger.info(f"  -> No improvement ({patience_counter}/{cfg.train.es_patience})")
        
        # Early stopping
        if cfg.train.early_stopping and patience_counter >= cfg.train.es_patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
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
            logger.info(f"  -> Periodic save: {ckpt_path}")
    
    # Stage summary
    stage_time = time.time() - stage_start_time
    mem = get_gpu_memory_info()
    logger.info("-" * 40)
    logger.info(f"Training Complete")
    logger.info(f"  Best Dice: {best_dice:.4f}")
    logger.info(f"  Total Epochs: {epoch+1}")
    logger.info(f"  Total Time: {format_time(stage_time)}")
    logger.info(f"  Max GPU Memory: {mem['max_allocated']:.2f} GB")
    logger.info("-" * 40)
    
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
    logger.info("=" * 60)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 60)
    
    # Load best weights
    if os.path.exists(cfg.eval.checkpoint):
        ckpt = torch.load(cfg.eval.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded weights from {cfg.eval.checkpoint}")
        if 'epoch' in ckpt:
            logger.info(f"  Checkpoint from epoch: {ckpt['epoch']+1}")
        if 'best_dice' in ckpt:
            logger.info(f"  Val Dice at save: {ckpt['best_dice']:.4f}")
    else:
        logger.warning(f"No checkpoint found at {cfg.eval.checkpoint}")
    
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
    eval_start = time.time()
    metrics = evaluate(model, test_loader, device, cfg.eval.threshold)
    eval_time = time.time() - eval_start
    
    mem = get_gpu_memory_info()
    
    logger.info("-" * 40)
    logger.info(f"Test Results:")
    logger.info(f"  Dice: {metrics['dice']*100:.2f}%")
    logger.info(f"  IoU:  {metrics['iou']*100:.2f}%")
    logger.info(f"  Time: {format_time(eval_time)}")
    logger.info(f"  GPU Memory: {mem['allocated']:.2f} GB")
    logger.info("-" * 40)
    
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
    
    logger = setup_logger(cfg.logging.log_dir, cfg.exp_name, args.mode)
    
    # Log start time and config
    start_time = time.time()
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Mode: {args.mode}")
    
    # Log full config
    log_config(cfg, logger)
    
    # Build model
    model = build_model(cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} (trainable: {trainable_params:,})")
    
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
    
    # Final summary
    total_time = time.time() - start_time
    mem = get_gpu_memory_info()
    logger.info("=" * 60)
    logger.info("Experiment Complete")
    logger.info("=" * 60)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info(f"Total Time: {format_time(total_time)}")
    logger.info(f"Peak GPU Memory: {mem['max_allocated']:.2f} GB")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
