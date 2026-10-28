"""
IRIS - Training and Evaluation Script
"""
import argparse
import os
import logging
import time
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import IRIS, DeepSupervisionLoss, CaCLLoss, DiceBCELoss
from models.iris_medklip import (
    IRISMedKLIP, build_iris_medklip, 
    IRISAblation, build_iris_ablation,
    IRISAblationNative, build_iris_ablation_native,
    IRISAblationWithAdapter, build_iris_ablation_with_adapter
)
from datasets import get_dataloader
from utils import (
    compute_metrics, set_seed, load_config, get_device, 
    save_checkpoint, load_checkpoint
)


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


def setup_logger(log_file: str, append: bool = False) -> logging.Logger:
    """Setup logging to file and console.
    
    Args:
        log_file: Path to log file
        append: If True, append to existing log file. If False, overwrite.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('iris')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(console)
    
    # File handler - append ('a') or overwrite ('w')
    file_mode = 'a' if append else 'w'
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Write header with timestamp
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"IRIS")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")
    
    return logger


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build IRIS or IRIS-MedKLIP model from config."""
    mcfg = cfg['model']
    backbone = mcfg.get('backbone', 'clip')
    
    if backbone == 'ablation_adapter':
        # Build IRIS ablation with adapters (frozen MedKLIP + trainable adapters)
        model = build_iris_ablation_with_adapter(cfg)
    elif backbone == 'ablation_native':
        # Build IRIS ablation model with native ResNet50 channels (no adapters)
        model = build_iris_ablation_native(cfg)
    elif backbone == 'ablation':
        # Build IRIS ablation model with flexible encoder selection
        model = build_iris_ablation(cfg)
    elif backbone == 'medklip':
        # Build IRIS with MedKLIP backbone (both visual + text)
        model = IRISMedKLIP(
            num_classes=mcfg['num_classes'],
            img_size=cfg['data']['image_size'],
            medklip_visual_path=mcfg.get('medklip_visual_path'),
            medklip_text_model=mcfg.get('medklip_text_model', 
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
            max_token_len=mcfg['max_token_len'],
            base_channel=mcfg['base_channel'],
            num_heads=mcfg['num_heads'],
            dropout=mcfg['dropout'],
            expansion=mcfg['expansion'],
            deep_supervision=mcfg['deep_supervision'],
            freeze_visual=mcfg.get('freeze_medklip_visual', False),
            freeze_text=mcfg.get('freeze_medklip_text', True),
            use_entity_translation=mcfg.get('use_entity_translation', True),
            text_mask_rate=mcfg['text_mask_rate'],
            img_mask_rate=mcfg['img_mask_rate'],
        )
    else:
        # Default: Build IRIS with CLIP backbone
        model = IRIS(
            num_classes=mcfg['num_classes'],
            img_size=cfg['data']['image_size'],
            clip_model=mcfg['clip_model'],
            max_token_len=mcfg['max_token_len'],
            base_channel=mcfg['base_channel'],
            num_heads=mcfg['num_heads'],
            dropout=mcfg['dropout'],
            expansion=mcfg['expansion'],
            encoder_type=mcfg['encoder_type'],
            deep_supervision=mcfg['deep_supervision'],
            text_mask_rate=mcfg['text_mask_rate'],
            img_mask_rate=mcfg['img_mask_rate'],
        )
    
    return model.to(device)


# -----------------------------------------------------------------------------
# Stage I: CaCL Pretraining
# -----------------------------------------------------------------------------
def pretrain(cfg: dict, model: nn.Module, device: torch.device, logger: logging.Logger):
    """Stage I: Category-aware Contrastive Learning."""
    pcfg = cfg['pretrain']
    
    # Dump full config to log
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    for section, values in cfg.items():
        if isinstance(values, dict):
            logger.info(f"[{section}]")
            for k, v in values.items():
                logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
    # Dataloader
    train_loader = get_dataloader(
        cfg['data']['train_dir'],
        batch_size=pcfg['batch_size'],
        image_size=cfg['data']['image_size'],
        augment=True,
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
    )
    
    # Loss
    criterion = CaCLLoss(temperature=pcfg['temperature'], eps=pcfg['eps'])
    
    # Optimizer - only train visual encoder + projections, language encoder is frozen
    params = [p for n, p in model.named_parameters()
              if 'lang_enc' not in n and p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    optimizer = torch.optim.Adam(params, lr=pcfg['lr'], weight_decay=pcfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=pcfg['epochs'])
    
    # Checkpoint dir
    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Dataset: {len(train_loader.dataset)} samples | Batch size: {pcfg['batch_size']} | Steps/epoch: {len(train_loader)}")
    logger.info(f"Config: τ={pcfg['temperature']}, ρ={pcfg['eps']}, lr={pcfg['lr']}")
    logger.info("-" * 60)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(pcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{pcfg['epochs']}", leave=False)
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
            f"Epoch [{epoch+1:02d}/{pcfg['epochs']}] "
            f"Loss: {loss_meter.avg:.4f} | "
            f"LR: {lr_current:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            save_checkpoint(
                {'epoch': epoch+1, 'model': model.state_dict(), 'best_loss': best_loss},
                cfg['pretrain']['checkpoint']
            )
    
    logger.info("-" * 60)
    logger.info(f"Pretrain done. Best loss: {best_loss:.4f}")
    logger.info(f"Saved to {cfg['pretrain']['checkpoint']}")
    
    return model


# -----------------------------------------------------------------------------
# Stage II: Segmentation Training
# -----------------------------------------------------------------------------
def train(cfg: dict, model: nn.Module, device: torch.device, logger: logging.Logger):
    """Stage II: Segmentation Training with Deep Supervision."""
    tcfg = cfg['train']
    mcfg = cfg['model']
    
    # Dump full config to log
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    for section, values in cfg.items():
        if isinstance(values, dict):
            logger.info(f"[{section}]")
            for k, v in values.items():
                logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
    # Load pretrained weights if available
    pretrain_ckpt = cfg['pretrain']['checkpoint']
    if cfg['pretrain']['enabled'] and Path(pretrain_ckpt).exists():
        load_checkpoint(pretrain_ckpt, model)
        logger.info(f"Loaded pretrained weights from {pretrain_ckpt}")
    elif cfg['pretrain']['enabled']:
        logger.warning("Pretrain enabled but no checkpoint found, training from scratch")
    else:
        logger.info("Pretrain disabled, training from scratch")
    
    # Dataloaders
    train_loader = get_dataloader(
        cfg['data']['train_dir'],
        batch_size=tcfg['batch_size'],
        image_size=cfg['data']['image_size'],
        augment=True,
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
    )
    
    val_loader = get_dataloader(
        cfg['data']['val_dir'],
        batch_size=tcfg['batch_size'],
        image_size=cfg['data']['image_size'],
        augment=False,
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )
    
    # Loss
    if mcfg['deep_supervision']:
        criterion = DeepSupervisionLoss(
            loss_weights=tcfg['loss_weights'],
            asymmetric_weights=tcfg['asymmetric_weights'],
            bce_weight=tcfg['bce_weight'],
        )
        logger.info(f"Using Deep Supervision Loss with weights: {tcfg['loss_weights']}")
    else:
        criterion = DiceBCELoss(bce_weight=tcfg['bce_weight'])
        logger.info("Using DiceBCE Loss")
    
    # Optimizer with different LR for visual encoder
    backbone = mcfg.get('backbone', 'clip')
    if backbone == 'medklip':
        # MedKLIP: visual_encoder is the ResNet50
        visual_encoder_prefixes = ('visual_encoder',)
    elif backbone in ('ablation', 'ablation_native', 'ablation_adapter'):
        # Ablation: depends on visual_encoder_type
        visual_type = mcfg.get('visual_encoder', 'cnn')
        if visual_type == 'medklip' or backbone in ('ablation_native', 'ablation_adapter'):
            visual_encoder_prefixes = ('visual_encoder',)
        else:
            visual_encoder_prefixes = ('down0', 'down1', 'down2', 'down3', 'pool')
    else:
        # CLIP: original encoder structure
        visual_encoder_prefixes = ('down0', 'down1', 'down2', 'down3', 'down4', 'pool')
    
    # Text encoder prefixes to exclude (frozen)
    text_encoder_prefixes = ('lang_enc', 'text_encoder')
    
    visual_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(prefix in n for prefix in text_encoder_prefixes):
            continue  # frozen
        if any(n.startswith(prefix) for prefix in visual_encoder_prefixes):
            visual_params.append(p)
        else:
            other_params.append(p)
    
    optimizer = torch.optim.Adam([
        {'params': visual_params, 'lr': tcfg['lr'] * 0.1},  # small lr for visual encoder
        {'params': other_params,  'lr': tcfg['lr']},
    ], weight_decay=tcfg['weight_decay'])
    
    # Scheduler
    if tcfg['cosine_lr']:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
        logger.info("Using CosineAnnealingWarmRestarts scheduler (T_0=10, eta_min=1e-4)")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg['epochs'])
    
    # Checkpoint dir
    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")
    logger.info(f"Batch size: {tcfg['batch_size']} | Steps/epoch: {len(train_loader)}")
    logger.info("-" * 60)
    
    # Training loop
    best_dice = 0.0
    no_improve = 0
    
    for epoch in range(tcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['epochs']}", leave=False)
        for batch_idx, (images, masks, texts) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)  # [B, 1, H, W]
            
            # Forward
            outputs = model(images, texts)
            
            # Compute loss
            if mcfg['deep_supervision'] and 'aux_preds' in outputs:
                loss = criterion(outputs['pred'], outputs['aux_preds'], masks)
            else:
                loss = criterion(outputs['pred'], masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            if tcfg['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
            
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{loss_meter.avg:.4f}'})
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Validation
        metrics = evaluate(model, val_loader, device, cfg['eval']['threshold'])
        
        logger.info(
            f"Epoch [{epoch+1}/{tcfg['epochs']}] Loss: {loss_meter.avg:.4f} | "
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
        
        # Early stopping
        if tcfg['early_stopping'] and no_improve >= tcfg['es_patience']:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {no_improve} epochs)")
            break
    
    logger.info("-" * 60)
    logger.info(f"Training done. Best Dice: {best_dice:.2f}%")
    
    return model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
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


def full_evaluate(cfg: dict, model: nn.Module, device: torch.device, logger: logging.Logger):
    """Full evaluation on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    # Load best weights
    ckpt_path = cfg['eval']['checkpoint']
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
        cfg['data']['test_dir'],
        batch_size=cfg['train']['batch_size'],
        image_size=cfg['data']['image_size'],
        augment=False,
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    metrics = evaluate(model, test_loader, device, cfg['eval']['threshold'])
    
    logger.info("-" * 60)
    logger.info(f"Test Results | Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}%")
    logger.info("-" * 60)
    
    return metrics


# -----------------------------------------------------------------------------
# CLI & Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='IRIS Training')
    p.add_argument('--config', required=True, help='Config file (configs/qata.yaml)')
    p.add_argument('--tag', required=True, help='Experiment tag')
    p.add_argument('--mode', default='full', choices=['pretrain', 'train', 'eval', 'full'])
    
    # Overrides
    p.add_argument('--backbone', choices=['clip', 'medklip', 'ablation', 'ablation_native', 'ablation_adapter'], help='Backbone type')
    p.add_argument('--encoder_type', choices=['cnn', 'mix'])
    p.add_argument('--no_pretrain', action='store_true')
    p.add_argument('--checkpoint', help='Override checkpoint for eval')
    p.add_argument('--output_dir', default='output')
    
    # MedKLIP specific
    p.add_argument('--medklip_visual_path', help='Path to MedKLIP pretrained visual weights')
    p.add_argument('--medklip_text_model', help='HuggingFace model name for MedKLIP text encoder')
    
    # Ablation specific
    p.add_argument('--visual_encoder', choices=['cnn', 'medklip'], help='Visual encoder type (ablation mode)')
    p.add_argument('--text_encoder', choices=['clip', 'medklip'], help='Text encoder type (ablation mode)')
    p.add_argument('--fusion_type', choices=['bcam', 'bifmd'], help='Fusion module type (ablation mode)')
    p.add_argument('--freeze_visual', action='store_true', help='Freeze visual encoder (MedKLIP)')
    p.add_argument('--freeze_text', action='store_true', help='Freeze text encoder')
    
    # Adapter specific
    p.add_argument('--adapter_type', choices=['bottleneck', 'spatial', 'multiscale'], 
                   default='bottleneck', help='Adapter type (for ablation_adapter backbone)')
    p.add_argument('--adapter_reduction', type=int, default=4, help='Adapter channel reduction ratio')
    p.add_argument('--adapter_scale', type=float, default=0.1, help='Initial adapter scale')
    
    # Training overrides
    p.add_argument('--epochs', type=int, help='Override number of training epochs')
    p.add_argument('--batch_size', type=int, help='Override batch size')
    p.add_argument('--max_token_len', type=int, help='Override max token length')
    
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Apply CLI overrides
    if args.backbone:
        cfg['model']['backbone'] = args.backbone
    if args.encoder_type:
        cfg['model']['encoder_type'] = args.encoder_type
    if args.no_pretrain:
        cfg['pretrain']['enabled'] = False
    
    # MedKLIP specific overrides
    if args.medklip_visual_path:
        cfg['model']['medklip_visual_path'] = args.medklip_visual_path
    if args.medklip_text_model:
        cfg['model']['medklip_text_model'] = args.medklip_text_model
    
    # Ablation specific overrides
    if args.visual_encoder:
        cfg['model']['visual_encoder'] = args.visual_encoder
    if args.text_encoder:
        cfg['model']['text_encoder'] = args.text_encoder
    if args.fusion_type:
        cfg['model']['fusion_type'] = args.fusion_type
    if args.freeze_visual:
        cfg['model']['freeze_medklip_visual'] = True
    if args.freeze_text:
        cfg['model']['freeze_medklip_text'] = True
    
    # Adapter overrides
    if args.adapter_type:
        cfg['model']['adapter_type'] = args.adapter_type
    if args.adapter_reduction:
        cfg['model']['adapter_reduction'] = args.adapter_reduction
    if args.adapter_scale:
        cfg['model']['adapter_scale'] = args.adapter_scale
    
    # Training overrides
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
    if args.max_token_len:
        cfg['model']['max_token_len'] = args.max_token_len
    
    # Set experiment name and paths
    dataset = cfg['project']['dataset']
    exp_name = f"{dataset}_{args.tag}"
    cfg['exp_name'] = exp_name
    
    output_dir = args.output_dir
    cfg['logging']['log_file'] = f"{output_dir}/logs/{exp_name}.log"
    cfg['logging']['checkpoint_dir'] = f"{output_dir}/checkpoints/{exp_name}"
    cfg['pretrain']['checkpoint'] = f"{output_dir}/checkpoints/{exp_name}/pretrain.pth"
    cfg['eval']['checkpoint'] = f"{output_dir}/checkpoints/{exp_name}/best.pth"
    
    if args.checkpoint:
        cfg['eval']['checkpoint'] = args.checkpoint
    
    # Setup
    set_seed(cfg['project']['seed'])
    device = get_device()
    
    os.makedirs(cfg['logging']['checkpoint_dir'], exist_ok=True)
    
    # Use append mode for eval-only to preserve training logs
    append_log = (args.mode == 'eval')
    logger = setup_logger(cfg['logging']['log_file'], append=append_log)
    
    # Log experiment info
    logger.info(f"Experiment: {cfg['exp_name']}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")
    
    # Build model
    model = build_model(cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Run
    if args.mode == 'pretrain':
        if cfg['pretrain']['enabled']:
            pretrain(cfg, model, device, logger)
        else:
            logger.info("Pretrain is disabled in config, skipping")
        
    elif args.mode == 'train':
        train(cfg, model, device, logger)
        
    elif args.mode == 'eval':
        full_evaluate(cfg, model, device, logger)
        
    elif args.mode == 'full':
        if cfg['pretrain']['enabled']:
            pretrain(cfg, model, device, logger)
        else:
            logger.info("Pretrain is disabled, skipping Stage I")
        train(cfg, model, device, logger)
        full_evaluate(cfg, model, device, logger)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
