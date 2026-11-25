"""
MRSeg — Training Script
Adapted from iris/run.py
"""
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
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
    logger.info("=" * 60)
    logger.info("MRSeg")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    return logger


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_dice, all_iou = [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        texts  = batch["text"]
        out    = model(images, texts)
        pred   = out["pred"]
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
def train(cfg, model, device, logger, resume=None):
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    logger.info("Configuration:")
    for sec, vals in cfg.items():
        if isinstance(vals, dict):
            logger.info(f"  [{sec}]")
            for k, v in vals.items():
                logger.info(f"    {k}: {v}")

    train_loader = get_dataloader(
        dcfg["train_dir"], batch_size=tcfg["batch_size"],
        image_size=dcfg["image_size"], augment=True, shuffle=True,
        num_workers=dcfg.get("num_workers", 4),
    )
    val_loader = get_dataloader(
        dcfg["val_dir"], batch_size=tcfg["batch_size"],
        image_size=dcfg["image_size"], augment=False, shuffle=False,
        num_workers=dcfg.get("num_workers", 4),
    )

    criterion = DiceBCELoss(bce_weight=1.0)

    # Param groups: trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 1e-4)
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"], eta_min=tcfg["lr"] * 1e-2
    )

    trainable = sum(p.numel() for p in params)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / Total: {total:,}")
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    logger.info("-" * 60)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice  = 0.0
    no_improve = 0
    start_epoch = 0
    patience   = tcfg.get("patience", 60)

    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_dice   = ckpt["best_dice"]
        no_improve  = ckpt["no_improve"]
        logger.info(f"Resumed from {resume} at epoch {start_epoch} (best Dice={best_dice:.2f}%)")

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['epochs']}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            texts  = batch["text"]
            out    = model(images, texts)
            loss   = criterion(out["pred"], masks)
            optimizer.zero_grad()
            loss.backward()
            if tcfg.get("grad_clip", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            optimizer.step()
            loss_m.update(loss.item(), images.size(0))
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        logger.info(
            f"[{epoch+1:3d}/{tcfg['epochs']}] loss={loss_m.avg:.4f} | "
            f"Dice={metrics['dice']:.2f}% | IoU={metrics['iou']:.2f}% | "
            f"{time.time()-t0:.0f}s"
        )

        if metrics["dice"] > best_dice:
            best_dice  = metrics["dice"]
            no_improve = 0
            save_checkpoint(
                {"epoch": epoch+1, "model": model.state_dict(), "best_dice": best_dice},
                str(ckpt_dir / "best.pth"),
            )
            logger.info(f"  → Saved best (Dice={best_dice:.2f}%)")
        else:
            no_improve += 1

        save_checkpoint(
            {"epoch": epoch+1, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "best_dice": best_dice, "no_improve": no_improve},
            str(ckpt_dir / "last.pth"),
        )
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Training done. Best Dice: {best_dice:.2f}%")
    return model


def full_evaluate(cfg, model, device, logger):
    logger.info("=" * 60 + "\nEVALUATION\n" + "=" * 60)
    ckpt_path = cfg.get("checkpoint_path", "")
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model)
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
    p.add_argument("--resume",  default=None,   help="train: last.pth to resume from; eval: checkpoint to evaluate")
    p.add_argument("--lr",      type=float,     help="Override learning rate")
    p.add_argument("--epochs",  type=int,       help="Override epochs")
    p.add_argument("--batch",   type=int,       help="Override batch size")
    p.add_argument("--output",  default="output", help="Output root dir")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.lr:     cfg["train"]["lr"]         = args.lr
    if args.epochs: cfg["train"]["epochs"]      = args.epochs
    if args.batch:  cfg["train"]["batch_size"]  = args.batch

    # Paths
    dataset  = cfg.get("project", {}).get("dataset", "mrseg")
    exp_name = f"{dataset}_{args.tag}"
    cfg["exp_name"]        = exp_name
    cfg["checkpoint_dir"]  = f"{args.output}/checkpoints/{exp_name}"
    # train: --resume continues training (from last.pth); final test always uses best.pth
    # eval : --resume is the checkpoint to evaluate
    best_path = f"{cfg['checkpoint_dir']}/best.pth"
    cfg["checkpoint_path"] = args.resume if (args.mode == "eval" and args.resume) else best_path
    log_file               = f"{args.output}/logs/{exp_name}.log"

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    logger = setup_logger(log_file, append=(args.mode == "eval"))
    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Device     : {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU        : {torch.cuda.get_device_name(0)}")

    model = build_mrseg(cfg).to(device)
    logger.info(f"Model built. Total params: {sum(p.numel() for p in model.parameters()):,}")

    if args.mode == "train":
        train(cfg, model, device, logger, resume=args.resume)
        full_evaluate(cfg, model, device, logger)
    else:
        full_evaluate(cfg, model, device, logger)

    logger.info(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
