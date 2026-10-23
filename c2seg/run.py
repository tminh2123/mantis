"""
Training script for C2Seg.
Usage:
  # QaTa-COV19 dataset
  python run.py --stage train --config configs/qata.yaml --tag qata_exp1
  
  # MosMedData+ dataset  
  python run.py --stage train --config configs/mosmed.yaml --tag mosmed_exp1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import set_seed, get_device, save_checkpoint, load_checkpoint, load_config
from utils.logger import get_logger
from utils.metrics import AverageMeter
from utils.seg_metrics import dice_score, miou_score, hd95, assd
from models.c2seg import C2Seg
from models.losses import CaCLLoss, BCEDiceLoss, DeepSupervisionLoss, WeightedDiceBCE
from datasets.dataset import C2SegDataset


# ---------------------------------------------------------------------------
# Stage I: CaCL pretraining
# ---------------------------------------------------------------------------
def pretrain(cfg):
    import time
    import json
    from tqdm import tqdm

    logger = get_logger('c2seg_pretrain', cfg['logging']['log_dir'])
    device = get_device()
    set_seed(cfg['project']['seed'])
    pcfg = cfg['pretrain']

    # Dump full config to log
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    for section, values in cfg.items():
        logger.info(f"[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {values}")
    logger.info("=" * 60)

    dataset = C2SegDataset(cfg['data']['train_dir'],
                           image_size=cfg['data']['image_size'],
                           augment=True)
    loader  = DataLoader(dataset, batch_size=pcfg['batch_size'],
                         shuffle=True, num_workers=cfg['data']['num_workers'],
                         pin_memory=True, drop_last=True)

    model = C2Seg(**cfg['model']).to(device)
    cacl  = CaCLLoss(temperature=pcfg['temperature'], eps=pcfg['eps'])

    # only train visual encoder + projections, language encoder is frozen
    params = [p for n, p in model.named_parameters()
              if 'lang_enc' not in n and p.requires_grad]
    trainable_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    optimizer = torch.optim.Adam(params, lr=pcfg['lr'],
                                 weight_decay=pcfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pcfg['epochs'])

    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {len(dataset)} samples | Batch size: {pcfg['batch_size']} | Steps/epoch: {len(loader)}")
    logger.info(f"Config: τ={pcfg['temperature']}, ρ={pcfg['eps']}, lr={pcfg['lr']}")
    logger.info("-" * 60)

    best_loss = float('inf')
    for epoch in range(pcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        epoch_start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{pcfg['epochs']}", leave=False)
        for batch_idx, (images, _, texts) in enumerate(pbar):
            images = images.to(device)

            # single forward pass — v, l projected; l_raw from frozen CLIP (no grad)
            v, l, l_raw = model.get_contrastive_embeddings(images, texts)

            loss = cacl(v, l, l_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))

            # Update progress bar
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
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict(), 'best_loss': best_loss},
                            str(ckpt_dir / 'c2seg_pretrain_best.pth'))

        # Save checkpoint periodically
        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict()},
                            str(ckpt_dir / f'c2seg_pretrain_epoch{epoch+1}.pth'))

    # Save final checkpoint
    save_checkpoint({'epoch': pcfg['epochs'], 'model': model.state_dict(), 'best_loss': best_loss},
                    pcfg['checkpoint'])
    logger.info("-" * 60)
    logger.info(f"Pretrain done. Best loss: {best_loss:.4f}")
    logger.info(f"Saved to {pcfg['checkpoint']}")
    return model


# ---------------------------------------------------------------------------
# Stage II: segmentation fine-tuning
# ---------------------------------------------------------------------------
def train(cfg, pretrained_ckpt: str = None):
    import time
    from tqdm import tqdm
    
    logger = get_logger('c2seg_train', cfg['logging']['log_dir'])
    device = get_device()
    set_seed(cfg['project']['seed'])
    tcfg = cfg['train']

    # Dump full config to log
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    for section, values in cfg.items():
        logger.info(f"[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"  {values}")
    logger.info("=" * 60)

    train_ds = C2SegDataset(cfg['data']['train_dir'],
                            image_size=cfg['data']['image_size'], augment=True)
    val_ds   = C2SegDataset(cfg['data']['val_dir'],
                            image_size=cfg['data']['image_size'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'],
                              shuffle=True,  num_workers=cfg['data']['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['eval']['batch_size'],
                              shuffle=False, num_workers=cfg['data']['num_workers'])

    model = C2Seg(**cfg['model']).to(device)

    # load pretrained weights if available
    ckpt = pretrained_ckpt or cfg['pretrain']['checkpoint']
    if Path(ckpt).exists():
        load_checkpoint(ckpt, model)
        logger.info(f"Loaded pretrained weights from {ckpt}")

    # Use deep supervision loss if enabled (matching c2seg-origin)
    deep_sup = cfg['model'].get('deep_supervision', False)
    if deep_sup:
        # Deep supervision with asymmetric weights from c2seg-origin
        loss_weights = tcfg.get('loss_weights', [0.5, 0.2, 0.15, 0.1, 0.05])
        asymmetric_weights = tcfg.get('asymmetric_weights', [
            (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)
        ])
        criterion = DeepSupervisionLoss(
            loss_weights=loss_weights,
            asymmetric_weights=asymmetric_weights
        )
        logger.info(f"Using Deep Supervision Loss with weights: {loss_weights}")
        logger.info(f"Asymmetric weights: {asymmetric_weights}")
    else:
        # Simple weighted loss for single output
        criterion = WeightedDiceBCE(weights=(0.5, 0.5))
        logger.info("Using WeightedDiceBCE Loss")

    # visual encoder uses smaller lr (fine-tune), rest uses full lr
    # Visual encoder includes: down0-down4, pool
    visual_encoder_prefixes = ('down0', 'down1', 'down2', 'down3', 'down4', 'pool')
    visual_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('lang_enc'):
            continue  # frozen, skip
        if any(n.startswith(prefix) for prefix in visual_encoder_prefixes):
            visual_params.append(p)
        else:
            other_params.append(p)
    optimizer = torch.optim.Adam([
        {'params': visual_params, 'lr': tcfg['lr'] * 0.1},   # small lr for visual encoder
        {'params': other_params,  'lr': tcfg['lr']},
    ], weight_decay=tcfg['weight_decay'])
    
    # Use CosineAnnealingWarmRestarts like c2seg-origin if configured
    use_cosine_wr = tcfg.get('cosine_lr', True)
    if use_cosine_wr:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-4
        )
        logger.info("Using CosineAnnealingWarmRestarts scheduler (T_0=10, eta_min=1e-4)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg['epochs'])

    ckpt_dir = Path(cfg['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    no_improve = 0
    
    # Gradient accumulation
    accum_steps = cfg.get('accum_steps', 1)
    if accum_steps > 1:
        logger.info(f"Gradient accumulation: {accum_steps} steps (effective batch = {tcfg['batch_size'] * accum_steps})")

    for epoch in range(tcfg['epochs']):
        model.train()
        loss_meter = AverageMeter()
        optimizer.zero_grad()
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['epochs']}", leave=False)
        for batch_idx, (images, masks, texts) in enumerate(pbar):
            images = images.to(device)
            masks  = masks.to(device).float()  # Convert long to float for loss computation

            outputs = model(images, texts)
            
            # Handle deep supervision output (tuple) vs single output
            if isinstance(outputs, tuple):
                # DeepSupervisionLoss expects (outputs_tuple, targets)
                loss = criterion(outputs, masks)
                logits = outputs[0]  # main output for logging
            else:
                # WeightedDiceBCE expects (targets, logits)
                loss = criterion(masks, outputs)
                logits = outputs
            
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            loss.backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if tcfg.get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
            
            loss_meter.update(loss.item() * accum_steps, images.size(0))
            pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}', 'avg': f'{loss_meter.avg:.4f}'})

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # validation
        metrics = evaluate(model, val_loader, device, cfg['eval']['threshold'])
        logger.info(
            f"Epoch [{epoch+1}/{tcfg['epochs']}] Loss: {loss_meter.avg:.4f} | "
            f"Dice: {metrics['dice']:.2f}% | mIoU: {metrics['miou']:.2f}% | "
            f"HD95: {metrics['hd95']:.2f} | ASSD: {metrics['assd']:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            no_improve = 0
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict(),
                             'best_dice': best_dice},
                            str(ckpt_dir / 'c2seg_best.pth'))
        else:
            no_improve += 1

        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict()},
                            str(ckpt_dir / f'c2seg_epoch{epoch+1}.pth'))

        # Early stopping
        if tcfg.get('early_stopping', False) and no_improve >= tcfg.get('es_patience', 50):
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {no_improve} epochs)")
            break

    logger.info(f"Training done. Best Dice: {best_dice:.2f}%")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, threshold=0.5):
    from tqdm import tqdm
    model.eval()
    dice_m = AverageMeter()
    miou_m = AverageMeter()
    all_hd95 = []
    all_assd = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=True)
        for images, masks, texts in pbar:
            images = images.to(device)
            masks  = masks.to(device).float()  # Convert long to float
            outputs = model(images, texts)
            
            # Handle deep supervision output (tuple) vs single output
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            B = images.size(0)
            dice_m.update(dice_score(logits, masks, threshold), B)
            miou_m.update(miou_score(logits, masks, threshold), B)
            
            # Compute HD95/ASSD per sample and collect
            h = hd95(logits, masks, threshold)
            a = assd(logits, masks, threshold)
            if h > 0:  # Only count if valid (non-empty masks)
                all_hd95.append(h)
            if a > 0:
                all_assd.append(a)
            
            pbar.set_postfix({'dice': f'{dice_m.avg:.2f}%', 'miou': f'{miou_m.avg:.2f}%'})

    return {
        'dice': dice_m.avg, 
        'miou': miou_m.avg,
        'hd95': float(np.mean(all_hd95)) if all_hd95 else 0.0, 
        'assd': float(np.mean(all_assd)) if all_assd else 0.0
    }


def eval_mode(cfg, checkpoint_path=None, use_cpu=False):
    logger = get_logger('c2seg_eval', cfg['logging']['log_dir'])
    device = torch.device('cpu') if use_cpu else get_device()
    logger.info(f"Using device: {device}")

    test_ds = C2SegDataset(cfg['data']['test_dir'],
                           image_size=cfg['data']['image_size'], augment=False)
    loader  = DataLoader(test_ds, batch_size=cfg['eval']['batch_size'],
                         shuffle=False, num_workers=cfg['data']['num_workers'])

    model = C2Seg(**cfg['model']).to(device)
    ckpt = checkpoint_path or cfg['eval']['checkpoint']
    load_checkpoint(ckpt, model)

    metrics = evaluate(model, loader, device, cfg['eval']['threshold'])
    logger.info(
        f"Test Results | Dice: {metrics['dice']:.2f}% | mIoU: {metrics['miou']:.2f}% | "
        f"HD95: {metrics['hd95']:.2f} | ASSD: {metrics['assd']:.2f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',  required=True, choices=['pretrain', 'train', 'eval'])
    p.add_argument('--config', default='configs/qata.yaml')
    p.add_argument('--pretrained', default=None, help='Path to pretrained ckpt for stage II')
    p.add_argument('--checkpoint', default=None, help='Path to checkpoint for eval (overrides tag)')
    p.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    p.add_argument('--lr', type=float, default=None, help='Override learning rate')
    p.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    p.add_argument('--tag', type=str, default=None, help='Experiment tag for checkpoint naming')
    p.add_argument('--accum-steps', type=int, default=1, help='Gradient accumulation steps')
    p.add_argument('--cpu', action='store_true', help='Force CPU mode')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args.config)
    
    # Model config from yaml (no ablation args)
    cfg['model']['use_text'] = True
    cfg['model']['use_bcam'] = True
    cfg['model']['use_kgate'] = True
    cfg['model']['use_kan'] = True
    cfg['model']['deep_supervision'] = True  # Always use deep supervision
    cfg['model']['base_channel'] = 64
    cfg['model']['channel_scale'] = 1.0
    
    # Pass image size to model
    cfg['model']['img_size'] = cfg['data']['image_size']
    
    # Pass other config params
    cfg['model']['max_token_len'] = cfg['model'].get('max_token_len', 18)
    cfg['model']['dropout'] = cfg['model'].get('dropout', 0.5)
    
    # Override config with command line args
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
        cfg['pretrain']['epochs'] = args.epochs
    if args.lr:
        cfg['train']['lr'] = args.lr
        cfg['pretrain']['lr'] = args.lr
    if args.batch_size:
        cfg['train']['batch_size'] = args.batch_size
        cfg['pretrain']['batch_size'] = args.batch_size
    
    # Gradient accumulation - auto-divide batch size if not explicitly set
    accum = getattr(args, 'accum_steps', 1)
    cfg['accum_steps'] = accum
    
    if accum > 1 and not args.batch_size:
        # Auto-divide: keep effective batch same, reduce actual batch
        cfg['train']['batch_size'] = cfg['train']['batch_size'] // accum
        cfg['pretrain']['batch_size'] = cfg['pretrain']['batch_size'] // accum
        print(f"Auto-adjusted batch_size: {cfg['train']['batch_size']} (effective: {cfg['train']['batch_size'] * accum})")
    
    # Apply experiment tag to checkpoint and log paths
    if args.tag:
        tag = args.tag
        # Append accum steps to tag if > 1
        accum = getattr(args, 'accum_steps', 1)
        if accum > 1:
            tag = f"{tag}_accum{accum}"
        cfg['logging']['checkpoint_dir'] = f"output/checkpoints/{tag}"
        cfg['logging']['log_dir'] = f"output/logs/{tag}"
        cfg['pretrain']['checkpoint'] = f"output/checkpoints/{tag}/c2seg_pretrain.pth"
        cfg['eval']['checkpoint'] = f"output/checkpoints/{tag}/c2seg_best.pth"
        print(f"Experiment tag: {tag}")
        print(f"Checkpoints: output/checkpoints/{tag}/")
        print(f"Logs: output/logs/{tag}/")
    
    # Log settings
    stage = args.stage
    
    print("=" * 60)
    print(f"Stage: {stage.upper()}")
    print("=" * 60)
    print(f"Config file: {args.config}")
    if args.tag:
        print(f"Experiment tag: {args.tag}")
    print("-" * 60)
    print("Data Config:")
    print(f"  train_dir: {cfg['data']['train_dir']}")
    print(f"  val_dir: {cfg['data']['val_dir']}")
    print(f"  test_dir: {cfg['data']['test_dir']}")
    print(f"  image_size: {cfg['data']['image_size']}")
    print("-" * 60)
    
    if stage == 'pretrain':
        print("Pretrain Config:")
        print(f"  epochs: {cfg['pretrain']['epochs']}")
        print(f"  batch_size: {cfg['pretrain']['batch_size']}")
        print(f"  lr: {cfg['pretrain']['lr']}")
        print(f"  checkpoint: {cfg['pretrain']['checkpoint']}")
    elif stage == 'train':
        print("Train Config:")
        print(f"  epochs: {cfg['train']['epochs']}")
        print(f"  batch_size: {cfg['train']['batch_size']}")
        print(f"  lr: {cfg['train']['lr']}")
        print(f"  accum_steps: {cfg['accum_steps']}")
        print(f"  effective_batch: {cfg['train']['batch_size'] * cfg['accum_steps']}")
        print(f"  checkpoint_dir: {cfg['logging']['checkpoint_dir']}")
    elif stage == 'eval':
        print("Eval Config:")
        print(f"  checkpoint: {cfg['eval']['checkpoint']}")
        print(f"  batch_size: {cfg['eval']['batch_size']}")
        print(f"  threshold: {cfg['eval']['threshold']}")
    print("=" * 60)

    if args.stage == 'pretrain':
        pretrain(cfg)
    elif args.stage == 'train':
        train(cfg, pretrained_ckpt=args.pretrained)
    elif args.stage == 'eval':
        eval_mode(cfg, checkpoint_path=args.checkpoint, use_cpu=args.cpu)
