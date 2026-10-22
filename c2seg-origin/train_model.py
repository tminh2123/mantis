# -*- coding: utf-8 -*-
import os
import random
from thop import profile
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import config
from datasets.vlmDataset import VlmDataset
from datasets.Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from nets.C2Seg import MBINet
from one_epoch import train_one_epoch, val_one_epoch
from utils.functions import CosineAnnealingWarmRestarts, WeightedDiceBCE, read_text
from utils.exp_process import save_checkpoint
from utils.deepsupervision import AsymmetricDiceBCE, PatchMaskGenerator


def worker_init_fn(worker_id: int):
    """Ensure reproducibility in dataloader workers."""
    random.seed(config.seed + worker_id)


def get_dataset(split: str, augment: bool = False):
    """Return dataset object for given split."""
    file_path = os.path.join(config.dataset_root, split, "preprocessed.pt")
    return VlmDataset(file_path, augment=augment)


def get_model(model_name: str, cfg):
    """Factory method to build model by name."""
    return MBINet(cfg.get_ViT_config(), cfg.n_channels, cfg.n_classes, deep_supervision=False)


def train(save_path: str, log):
    """Main training loop."""
    # ---------------- Data ---------------- #
    train_loader = DataLoader(
        get_dataset("Train_Folder", augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=24,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        get_dataset("Val_Folder", augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=24,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # ---------------- Model ---------------- #
    model = get_model(config.model_name, config)

    # Profile model FLOPs & Params
    dummy_img = torch.randn(2, 3, config.img_size, config.img_size)
    dummy_token = torch.randint(0, 1000, (2, config.token_len))
    dummy_mask = (dummy_token != 0).int()
    with torch.no_grad():
        tmp_model = deepcopy(model).cpu()
        flops, params = profile(tmp_model, inputs=(dummy_img, dummy_token, dummy_mask), verbose=False)
        del tmp_model
    log.info(f"Params: {params / 1e6:.2f} M, FLOPs: {flops / 1e9:.2f} G")

    model = model.cuda()
    if torch.cuda.is_available():
        log.info(f"Using {torch.cuda.device_count()} GPU(s)\n")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # ---------------- Loss & Optimizer ---------------- #
    weights = [[0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
    criterions = [WeightedDiceBCE(w) for w in weights]
    patch_gens = [PatchMaskGenerator(ps).cuda() for ps in [2, 4, 8, 16]]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4) if config.cosineLR else None

    # ---------------- Resume ---------------- #
    if config.resume:
        checkpoint = torch.load(os.path.join(save_path, f"models/best_model-{config.model_name}.pth.tar"))
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        max_dice = checkpoint["max_dice"]
        best_epoch = checkpoint["best_epoch"]
        log.info(f"Resumed from epoch {start_epoch}, best dice={max_dice:.4f} at epoch {best_epoch}\n")
    else:
        start_epoch, max_dice, best_epoch = 1, 0.0, 1

    # ---------------- TensorBoard ---------------- #
    writer = None
    if config.tensorboard:
        tb_dir = os.path.join(save_path, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)

    # ---------------- Training Loop ---------------- #
    model_path = os.path.join(save_path, "models")
    os.makedirs(model_path, exist_ok=True)
    for epoch in range(start_epoch, config.epochs + 1):
        # ---- Train ----
        model.train()
        train_one_epoch(train_loader, model, criterions, patch_gens, optimizer, epoch, log, writer=writer)

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_dice = val_one_epoch(val_loader, model, criterions, patch_gens, optimizer, epoch, log,
                                     writer=writer, scheduler=scheduler, max_dice=max_dice, best_epoch=best_epoch)

        # ---- Checkpoint ----
        if val_dice > max_dice:
            if epoch > config.save_after:
                log.info(f"\t   Dice improved from {max_dice:.4f} → {val_dice:.4f}. Saving model... ↑\n")
                save_checkpoint({"epoch": epoch,
                                 "best_model": True,
                                 "max_dice": val_dice,
                                 "best_epoch": epoch,
                                 "model": config.model_name,
                                 "state_dict": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict() if scheduler else None}, model_path)
            else:
                log.info(f"\t   Dice improved from {max_dice:.4f} → {val_dice:.4f} ↑\n")
            max_dice, best_epoch = val_dice, epoch
        else:
            es_count = epoch - best_epoch
            log.info(f"\t   Dice={val_dice:.4f}, best = {max_dice:.4f} at epoch {best_epoch} (ES {es_count}/{config.es_patience}) ↓\n")
            if es_count >= config.es_patience:
                log.info("\t   Early stopping triggered.")
                break

