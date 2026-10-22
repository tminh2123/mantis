# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
from colorama import Fore
import config
from utils.functions import dice_on_batch, iou_on_batch


def _run_one_epoch(loader, model, criterions, patch_gens, optimizer, epoch, log, mode,
                   writer=None, scheduler=None, max_dice=None, best_epoch=None):
    """
    Run a single epoch (Train or Validation).
    """
    start_time = time.time()

    # --- Initialize sums ---
    loss_sum, dice_sum, iou_sum = 0.0, 0.0, 0.0
    loss_avg = dice_avg = iou_avg = elapsed = 0.0

    lr = min(g["lr"] for g in optimizer.param_groups)
    bar = tqdm(loader, desc=f"{Fore.CYAN}[{mode} {epoch}]", ncols=150, leave=True)

    for i, (sample, _) in enumerate(bar, 1):
        # Move inputs to GPU
        image, mask, text_token, text_mask = (
            sample["image"].cuda(), sample["mask"].cuda(), sample["text_token"].cuda(), sample["text_mask"].cuda()
        )

        # Generate patch masks
        patch_masks = [gen(mask.unsqueeze(1)) for gen in patch_gens]

        # Forward pass
        masks = [mask] + patch_masks
        outs = model(image, text_token, text_mask)
        losses = [c(m, o) for m, o, c in zip(masks, outs, criterions)]
        out_loss = sum(w * l for w, l in zip(config.loss_weight, losses))

        if mode == "Train":
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        # Metrics
        dice_main = dice_on_batch(mask, outs[0])
        iou_main = iou_on_batch(mask, outs[0])

        # Accumulate
        loss_sum += out_loss.item() * len(image)
        dice_sum += dice_main * len(image)
        iou_sum += iou_main * len(image)

        # Running averages
        total_seen = (i - 1) * loader.batch_size + len(image)
        loss_avg = loss_sum / total_seen
        dice_avg = dice_sum / total_seen
        iou_avg = iou_sum / total_seen

        # Progress bar message
        elapsed = time.time() - start_time
        if max_dice is not None:
            if dice_avg > max_dice:
                msg_dice = f"{Fore.GREEN} Dice:{dice_avg:.4f}"
            else:
                msg_dice = f"{Fore.RED} Dice:{dice_avg:.4f}"
        else:
            msg_dice = f"{Fore.CYAN} Dice:{dice_avg:.4f}"
        msg = (f"{Fore.CYAN}Loss:{loss_avg:.4f}" + msg_dice + f"{Fore.CYAN} IoU:{iou_avg:.4f}")

        if mode == "Train":
            msg += f" LR:{lr:.2e}"
        elif i == len(loader) and max_dice is not None:
            if dice_avg > max_dice:
                if epoch > config.save_after:
                    msg += f"{Fore.GREEN} Best:{max_dice:.4f} ↑(Saving.){Fore.CYAN}  "
                else:
                    msg += f"{Fore.GREEN} Best:{max_dice:.4f} ↑(Waiting.){Fore.CYAN} "
            else:
                msg += f"{Fore.GREEN} Best:{max_dice:.4f}{Fore.RED} ↓(ES {epoch - best_epoch:2d}/{config.es_patience}){Fore.CYAN} "

        bar.set_postfix_str(msg)

    # --- Logging at epoch-level ---
    log.file()
    log.info(f"[{mode} {epoch}] Loss:{loss_avg:.4f} Dice:{dice_avg:.4f}"
             + f" IoU:{iou_avg:.4f} Time:{elapsed:.2f} s" + (f" LR:{lr:.2e} " if mode == "Train" else ""))

    # TensorBoard record once per epoch
    if config.tensorboard and writer:
        writer.add_scalar(f"{mode}_loss", loss_avg, epoch)
        writer.add_scalar(f"{mode}_dice", dice_avg, epoch)

    if mode == "Val":
        if scheduler is not None:
            scheduler.step()
        print()

    return dice_avg


def train_one_epoch(*args, **kwargs):
    return _run_one_epoch(*args, mode="Train", **kwargs)


def val_one_epoch(*args, **kwargs):
    return _run_one_epoch(*args, mode="Val", **kwargs)

