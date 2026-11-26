"""
MRSeg — Training Script (DDP + single-GPU)

Single GPU:
    python run.py --config configs/qata.yaml --tag v1 --gpu 0

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 run.py --config configs/qata.yaml --tag v1 --gpu 0,2
"""
import argparse
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.mrseg import build_mrseg
from datasets.dataset import MRSegDataset, get_dataloader
from utils import (
    compute_metrics_per_sample, set_seed, load_config, get_device,
    save_checkpoint, load_checkpoint, DiceBCELoss,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model, unwrapping DDP or DataParallel if needed."""
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


def setup_logger(log_file: str, append: bool = False) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("mrseg")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt_short = logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S")
    fmt_long  = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt_short); logger.addHandler(sh)
    fh = logging.FileHandler(log_file, mode="a" if append else "w")
    fh.setFormatter(fmt_long); logger.addHandler(fh)
    logger.info("")
    logger.info("=" * 60)
    logger.info("MRSeg")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")
    return logger


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    _model = unwrap_model(model)   # bypass DDP to avoid sync issues
    _model.eval()
    all_dice, all_iou = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        texts  = batch["text"]
        out  = _model(images, texts)
        pred = out["pred"]
        m = compute_metrics_per_sample(pred, masks, threshold)
        all_dice.append(m["dice"])
        all_iou.append(m["iou"])
    return {
        "dice": torch.cat(all_dice).mean().item() * 100,
        "iou":  torch.cat(all_iou).mean().item()  * 100,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg, model, device, logger, resume=None, is_main=True, world_size=1):
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    is_ddp = world_size > 1

    if is_main:
        logger.info("=" * 60)
        logger.info("CONFIGURATION")
        logger.info("=" * 60)
        for sec, vals in cfg.items():
            if isinstance(vals, dict):
                logger.info(f"[{sec}]")
                for k, v in vals.items():
                    logger.info(f"  {k}: {v}")
        logger.info("=" * 60)

    # DataLoaders
    per_gpu_batch = tcfg["batch_size"] // world_size
    train_dataset = MRSegDataset(dcfg["train_dir"], image_size=dcfg["image_size"], augment=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    # With DDP, forking DataLoader workers after NCCL init causes deadlock
    num_workers = 0 if is_ddp else dcfg.get("num_workers", 4)

    train_loader  = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val only on main process
    val_loader = None
    if is_main:
        val_loader = get_dataloader(
            dcfg["val_dir"], batch_size=per_gpu_batch,
            image_size=dcfg["image_size"], augment=False, shuffle=False,
            num_workers=dcfg.get("num_workers", 4),
        )

    criterion = DiceBCELoss(bce_weight=1.0)

    # Differential LR (optional, can be disabled by setting encoder_lr_scale=1.0)
    base_lr       = tcfg["lr"]
    encoder_scale = tcfg.get("encoder_lr_scale", 1.0)
    encoder_lr    = base_lr * encoder_scale
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    min_lr        = tcfg.get("min_lr", 1e-6)

    encoder_params, new_params = [], []
    for n, p in unwrap_model(model).named_parameters():
        if not p.requires_grad:
            continue
        if "visual_enc" in n or "text_enc" in n:
            encoder_params.append(p)
        else:
            new_params.append(p)

    # If encoder_lr_scale == 1.0, use single param group for simplicity
    if encoder_scale == 1.0:
        optimizer = torch.optim.AdamW(
            encoder_params + new_params,
            lr=base_lr,
            weight_decay=tcfg.get("weight_decay", 1e-4)
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": encoder_lr},
            {"params": new_params,     "lr": base_lr},
        ], weight_decay=tcfg.get("weight_decay", 1e-4))

    # Paper: CosineAnnealingLR to min_lr (1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"], eta_min=min_lr)

    if is_main:
        logger.info(f"LR: {base_lr:.1e} (encoder_scale={encoder_scale}) | min_lr={min_lr:.1e}")
        logger.info(f"Encoder LoRA params: {sum(p.numel() for p in encoder_params):,}")
        logger.info(f"New module params:   {sum(p.numel() for p in new_params):,}")
        logger.info(f"Total trainable:     {sum(p.numel() for p in encoder_params + new_params):,}")
        logger.info(f"Effective batch:     {tcfg['batch_size']} ({per_gpu_batch}/GPU × {world_size})")
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_loader.dataset) if val_loader else 'N/A'}")
        logger.info("-" * 60)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice   = 0.0
    no_improve  = 0
    start_epoch = 0
    patience    = tcfg.get("patience", 60)
    use_amp     = device.type == "cuda"
    scaler      = GradScaler(enabled=use_amp)

    if is_main:
        logger.info(f"AMP: {'enabled' if use_amp else 'disabled'}")

    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_dice   = ckpt["best_dice"]
        no_improve  = ckpt["no_improve"]
        if is_main:
            logger.info(f"Resumed from {resume} at epoch {start_epoch} (best Dice={best_dice:.2f}%)")

    for epoch in range(start_epoch, tcfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # ensures different shuffle per epoch

        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['epochs']}", leave=False,
                    disable=not is_main)

        for batch in pbar:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            texts  = batch["text"]

            with autocast(enabled=use_amp):
                out  = model(images, texts)
                loss = criterion(out["pred"], masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip", 1.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder_params + new_params, tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            loss_m.update(loss.item(), images.size(0))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Evaluate and save only on main process
        if is_main:
            metrics = evaluate(model, val_loader, device)
            cur_lr  = optimizer.param_groups[-1]["lr"]  # last group (or only group)
            logger.info(
                f"Epoch [{epoch+1}/{tcfg['epochs']}] Loss: {loss_m.avg:.4f} | "
                f"Dice: {metrics['dice']:.2f}% | IoU: {metrics['iou']:.2f}% | "
                f"LR: {cur_lr:.2e} | Time: {time.time()-t0:.1f}s"
            )

            if metrics["dice"] > best_dice:
                best_dice  = metrics["dice"]
                no_improve = 0
                save_checkpoint(
                    {"epoch": epoch+1, "model": unwrap_model(model).state_dict(),
                     "best_dice": best_dice},
                    str(ckpt_dir / "best.pth"),
                )
                logger.info(f"  → Saved best (Dice={best_dice:.2f}%)")
            else:
                no_improve += 1

            save_checkpoint(
                {"epoch": epoch+1, "model": unwrap_model(model).state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(),
                 "best_dice": best_dice, "no_improve": no_improve},
                str(ckpt_dir / "last.pth"),
            )
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Sync early-stopping decision across processes
        if is_ddp:
            stop = torch.tensor(int(no_improve >= patience), device=device)
            dist.broadcast(stop, src=0)
            dist.barrier()   # ensure all ranks sync before next epoch
            if stop.item():
                break

    if is_main:
        logger.info(f"Training done. Best Dice: {best_dice:.2f}%")
    return model


def full_evaluate(cfg, model, device, logger):
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    ckpt_path = cfg.get("checkpoint_path", "")
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, unwrap_model(model))
        logger.info(f"Loaded: {ckpt_path}")
    test_loader = get_dataloader(
        cfg["data"]["test_dir"], batch_size=cfg["train"]["batch_size"],
        image_size=cfg["data"]["image_size"], augment=False, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )
    m = evaluate(model, test_loader, device)
    logger.info(f"Test | Dice: {m['dice']:.2f}% | IoU: {m['iou']:.2f}%")
    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MRSeg Training")
    p.add_argument("--config",  required=True, help="YAML config file")
    p.add_argument("--tag",     required=True, help="Experiment tag")
    p.add_argument("--mode",    default="train", choices=["train", "eval"])
    p.add_argument("--resume",  default=None,   help="train: last.pth to resume; eval: checkpoint to evaluate")
    p.add_argument("--lr",      type=float,     help="Override learning rate")
    p.add_argument("--epochs",  type=int,       help="Override epochs")
    p.add_argument("--batch",   type=int,       help="Override total batch size")
    p.add_argument("--output",  default="output", help="Output root dir")
    p.add_argument("--gpu",     default=None,   help="GPU IDs, e.g. '0' or '0,2'")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.lr:     cfg["train"]["lr"]         = args.lr
    if args.epochs: cfg["train"]["epochs"]      = args.epochs
    if args.batch:  cfg["train"]["batch_size"]  = args.batch

    # DDP detection (set by torchrun)
    is_ddp     = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = 1

    if is_ddp:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()

    is_main = (local_rank == 0)

    # Device setup
    if args.gpu:
        gpu_ids = [int(g) for g in args.gpu.split(",")]
        device  = torch.device(f"cuda:{gpu_ids[local_rank]}")
        torch.cuda.set_device(gpu_ids[local_rank])
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    # Paths
    dataset  = cfg.get("project", {}).get("dataset", "mrseg")
    exp_name = f"{dataset}_{args.tag}"
    cfg["exp_name"]       = exp_name
    cfg["checkpoint_dir"] = f"{args.output}/checkpoints/{exp_name}"
    best_path             = f"{cfg['checkpoint_dir']}/best.pth"
    cfg["checkpoint_path"] = args.resume if (args.mode == "eval" and args.resume) else best_path
    log_file              = f"{args.output}/logs/{exp_name}.log"

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed + local_rank)   # different seed per process

    if is_main:
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Logger (main process only)
    logger = setup_logger(log_file, append=(args.mode == "eval")) if is_main \
             else logging.getLogger("mrseg")

    if is_main:
        logger.info(f"Experiment : {exp_name}")
        logger.info(f"Device     : {device}")
        logger.info(f"Mode       : {'DDP' if is_ddp else 'single-GPU'} × {world_size}")
        if torch.cuda.is_available():
            logger.info(f"GPU(s)     : {torch.cuda.get_device_name(device)}")

    model = build_mrseg(cfg).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)

    if is_main:
        num_params = sum(p.numel() for p in unwrap_model(model).parameters())
        logger.info(f"Encoders   : ConvNeXt-Tiny (LoRA) + BiomedBERT (LoRA)")
        logger.info(f"Parameters : {num_params:,}")
        logger.info("")

    if args.mode == "train":
        train(cfg, model, device, logger, resume=args.resume,
              is_main=is_main, world_size=world_size)
        if is_main:
            full_evaluate(cfg, model, device, logger)
    else:
        if is_main:
            full_evaluate(cfg, model, device, logger)

    if is_main:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
