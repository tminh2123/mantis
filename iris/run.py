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

from models import IRIS, build_iris, DeepSupervisionLoss, DiceBCELoss
from datasets import get_dataloader
from utils import (
    compute_metrics_per_sample, set_seed, load_config, get_device, 
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
    """Setup logging to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('iris')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(console)
    
    file_handler = logging.FileHandler(log_file, mode='a' if append else 'w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"IRIS")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")
    
    return logger


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    """Build IRIS model from config."""
    model = build_iris(cfg)
    return model.to(device)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train(cfg: dict, model: nn.Module, device: torch.device, logger: logging.Logger):
    """Segmentation Training."""
    tcfg = cfg['train']
    mcfg = cfg['model']
    
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    for section, values in cfg.items():
        if isinstance(values, dict):
            logger.info(f"[{section}]")
            for k, v in values.items():
                logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
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
    if mcfg.get('deep_supervision', True):
        criterion = DeepSupervisionLoss(
            loss_weights=tcfg['loss_weights'],
            asymmetric_weights=tcfg['asymmetric_weights'],
            bce_weight=tcfg['bce_weight'],
        )
    else:
        criterion = DiceBCELoss(bce_weight=tcfg['bce_weight'])
    
    # Optimizer - differential LR for visual encoder
    visual_lr_scale = tcfg.get('visual_lr_scale', 0.1)
    base_lr = tcfg['lr']
    visual_lr = base_lr * visual_lr_scale
    
    visual_params = []
    adapter_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 't_enc' in n:
            continue  # frozen
        if 'v_enc' in n or 'ch_adapt' in n:
            visual_params.append(p)
        elif 'v_adapters' in n:
            adapter_params.append(p)  # Adapters get full LR
        else:
            other_params.append(p)
    
    param_groups = [
        {'params': visual_params, 'lr': visual_lr, 'name': 'visual'},
        {'params': adapter_params, 'lr': base_lr, 'name': 'adapter'},
        {'params': other_params, 'lr': base_lr, 'name': 'other'},
    ]
    # Remove empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    optimizer = torch.optim.Adam(param_groups, weight_decay=tcfg['weight_decay'])
    
    logger.info(f"LR: visual={visual_lr:.1e} ({visual_lr_scale}x), adapter/other={base_lr:.1e}")
    logger.info(f"Visual params: {sum(p.numel() for p in visual_params):,}")
    logger.info(f"Adapter params: {sum(p.numel() for p in adapter_params):,}")
    logger.info(f"Other params: {sum(p.numel() for p in other_params):,}")
    
    if tcfg.get('cosine_lr', True):
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg['epochs'])
    
    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    logger.info("-" * 60)
    
    best_dice = 0.0
    no_improve = 0
    
    for epoch in range(tcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['epochs']}", leave=False)
        for images, masks, texts in pbar:
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)
            
            outputs = model(images, texts)
            
            if mcfg.get('deep_supervision', True) and 'aux_preds' in outputs:
                loss = criterion(outputs['pred'], outputs['aux_preds'], masks)
            else:
                loss = criterion(outputs['pred'], masks)
            
            optimizer.zero_grad()
            loss.backward()
            
            if tcfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
            
            optimizer.step()
            
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        metrics = evaluate(model, val_loader, device, cfg['eval']['threshold'])
        
        logger.info(
            f"Epoch [{epoch+1}/{tcfg['epochs']}] Loss: {loss_meter.avg:.4f} | "
            f"Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )
        
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            no_improve = 0
            save_checkpoint(
                {'epoch': epoch+1, 'model': model.state_dict(), 'best_dice': best_dice},
                str(ckpt_dir / 'best.pth')
            )
        else:
            no_improve += 1
        
        if tcfg.get('early_stopping', False) and no_improve >= tcfg.get('es_patience', 20):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training done. Best Dice: {best_dice:.2f}%")
    return model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    """Evaluate model with proper Macro averaging."""
    model.eval()
    
    all_dice = []
    all_iou = []
    
    for images, masks, texts in loader:
        images = images.to(device)
        masks = masks.to(device).float()
        
        outputs = model(images, texts)
        pred = outputs['pred']
        
        # Get per-sample metrics
        metrics = compute_metrics_per_sample(pred, masks.unsqueeze(1), threshold)
        all_dice.append(metrics['dice'])  # [B]
        all_iou.append(metrics['iou'])    # [B]
    
    # Concat all samples and compute true Macro mean
    all_dice = torch.cat(all_dice)  # [N_total]
    all_iou = torch.cat(all_iou)    # [N_total]
    
    return {
        'dice': all_dice.mean().item() * 100,
        'iou': all_iou.mean().item() * 100,
    }


def full_evaluate(cfg: dict, model: nn.Module, device: torch.device, logger: logging.Logger):
    """Full evaluation on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    ckpt_path = cfg['eval']['checkpoint']
    if Path(ckpt_path).exists():
        ckpt = load_checkpoint(ckpt_path, model)
        logger.info(f"Loaded: {ckpt_path}")
    
    test_loader = get_dataloader(
        cfg['data']['test_dir'],
        batch_size=cfg['train']['batch_size'],
        image_size=cfg['data']['image_size'],
        augment=False,
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
    )
    
    metrics = evaluate(model, test_loader, device, cfg['eval']['threshold'])
    
    logger.info(f"Test | Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}%")
    return metrics


# -----------------------------------------------------------------------------
# CLI & Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='IRIS Training')
    p.add_argument('--config', required=True, help='Config file')
    p.add_argument('--tag', required=True, help='Experiment tag')
    p.add_argument('--mode', default='train', choices=['train', 'eval'])
    
    # Model options
    p.add_argument('--visual_encoder', choices=['cnn', 'resnet'], help='Visual encoder')
    p.add_argument('--text_encoder', choices=['clip', 'pubmedbert', 'clinicalbert'], help='Text encoder')
    p.add_argument('--fusion_type', choices=['bcam', 'bifmd'], help='Fusion module')
    p.add_argument('--adapter', choices=['bottleneck', 'spatial', 'multiscale'], help='Adapter type')
    p.add_argument('--visual_pretrained', help='Pretrained visual weights path')
    p.add_argument('--freeze_visual', action='store_true', help='Freeze visual encoder')
    
    # Fusion tuning
    p.add_argument('--num_heads', type=int, help='Number of attention heads (default: 8)')
    p.add_argument('--attn_dropout', type=float, help='Attention dropout (default: 0.0)')
    p.add_argument('--expansion', type=int, help='MLP expansion ratio (default: 4)')
    
    # Decoder attention
    p.add_argument('--decoder_attention', action='store_true', help='Enable self-attention in decoder')
    p.add_argument('--decoder_attention_heads', type=int, help='Decoder attention heads (default: 4)')
    p.add_argument('--decoder_attention_levels', type=int, help='Number of decoder levels with attention (default: 2)')
    
    # Learning rate
    p.add_argument('--lr', type=float, help='Base learning rate')
    p.add_argument('--visual_lr_scale', type=float, help='Visual encoder LR scale (default: 0.1)')
    
    # Training options
    p.add_argument('--epochs', type=int, help='Training epochs')
    p.add_argument('--patience', type=int, help='Early stopping patience (default: 20)')
    p.add_argument('--batch_size', type=int, help='Batch size')
    p.add_argument('--max_token_len', type=int, help='Max token length')
    p.add_argument('--checkpoint', help='Checkpoint for eval')
    p.add_argument('--output_dir', default='output', help='Output directory')
    
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Apply CLI overrides
    if args.visual_encoder:
        cfg['model']['visual_encoder'] = args.visual_encoder
    if args.text_encoder:
        cfg['model']['text_encoder'] = args.text_encoder
    if args.fusion_type:
        cfg['model']['fusion_type'] = args.fusion_type
    if args.adapter:
        cfg['model']['adapter'] = args.adapter
    if args.visual_pretrained:
        cfg['model']['visual_pretrained'] = args.visual_pretrained
    if args.freeze_visual:
        cfg['model']['freeze_visual'] = True
    
    # Fusion tuning overrides
    if args.num_heads:
        cfg['model']['num_heads'] = args.num_heads
    if args.attn_dropout is not None:
        cfg['model']['attn_dropout'] = args.attn_dropout
    if args.expansion:
        cfg['model']['expansion'] = args.expansion
    
    # Decoder attention overrides
    if args.decoder_attention:
        cfg['model']['decoder_attention'] = True
    if args.decoder_attention_heads:
        cfg['model']['decoder_attention_heads'] = args.decoder_attention_heads
    if args.decoder_attention_levels:
        cfg['model']['decoder_attention_levels'] = args.decoder_attention_levels
    
    # Learning rate overrides
    if args.lr:
        cfg['train']['lr'] = args.lr
    if args.visual_lr_scale is not None:
        cfg['train']['visual_lr_scale'] = args.visual_lr_scale
    
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
    if args.patience:
        cfg['train']['es_patience'] = args.patience
    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
    if args.max_token_len:
        cfg['model']['max_token_len'] = args.max_token_len
    
    # Set paths
    dataset = cfg['project']['dataset']
    exp_name = f"{dataset}_{args.tag}"
    cfg['exp_name'] = exp_name
    
    output_dir = args.output_dir
    cfg['logging']['log_file'] = f"{output_dir}/logs/{exp_name}.log"
    cfg['logging']['checkpoint_dir'] = f"{output_dir}/checkpoints/{exp_name}"
    cfg['eval']['checkpoint'] = f"{output_dir}/checkpoints/{exp_name}/best.pth"
    
    if args.checkpoint:
        cfg['eval']['checkpoint'] = args.checkpoint
    
    # Setup
    set_seed(cfg['project']['seed'])
    device = get_device()
    
    os.makedirs(cfg['logging']['checkpoint_dir'], exist_ok=True)
    
    append_log = (args.mode == 'eval')
    logger = setup_logger(cfg['logging']['log_file'], append=append_log)
    
    logger.info(f"Experiment: {cfg['exp_name']}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Build model
    model = build_model(cfg, device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: visual={cfg['model'].get('visual_encoder', 'cnn')}, "
                f"text={cfg['model'].get('text_encoder', 'clip')}, "
                f"fusion={cfg['model'].get('fusion_type', 'bcam')}")
    logger.info(f"Parameters: {num_params:,}")
    logger.info("")
    
    # Run
    if args.mode == 'train':
        train(cfg, model, device, logger)
        full_evaluate(cfg, model, device, logger)
    elif args.mode == 'eval':
        full_evaluate(cfg, model, device, logger)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
