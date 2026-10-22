# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import time

import config
from train_model import get_dataset, get_model
from datasets.Load_Dataset import ValGenerator, ImageToImage2D
from utils.functions import read_text, dice_on_batch, iou_on_batch


def vis_and_save(model, image, text_token, text_mask, mask, vis_path, name):
    """
    Forward pass and visualization for a single sample.
    Saves prediction binary mask, Grad-CAM heatmap, and overlay visualization if enabled.
    """
    model.eval()
    with torch.no_grad():
        image, text_token, text_mask = image.cuda(), text_token.cuda(), text_mask.cuda()
        pred = model(image, text_token, text_mask)[0]

    dice = dice_on_batch(mask, pred)
    iou = iou_on_batch(mask, pred)

    if config.visualization:
        # Predict
        predict = (pred >= 0.5).astype(np.uint8).squeeze()

        # Original image
        orig_img = torch.squeeze(image, 0).cpu().numpy().transpose(1, 2, 0)
        orig_img = (orig_img * 255).astype(np.uint8)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Binary prediction
        Image.fromarray(predict * 255).save(os.path.join(vis_path, "predict_binary", f"{name}_dice{dice:.2f}.png"))

        # Grad-CAM heatmap
        from modules.gradcam import GradCAM
        target_layer = model.module.up0 if isinstance(model, nn.DataParallel) else model.up0
        gradcam = GradCAM(model, target_layer=target_layer)
        heatmap = gradcam.generate_cam(image, text_token, text_mask, target_class=0)
        cam_overlay = cv2.addWeighted(orig_img, 0.6, heatmap[..., ::-1], 0.4, 0)
        Image.fromarray(cam_overlay).save(os.path.join(vis_path, "predict_heatmap", f"{name}_dice{dice:.2f}.png"))
        gradcam.remove_hooks()

        # Mixed overlay: TP=green, FP=blue, FN=red
        mask2d = np.squeeze(mask.cpu().numpy())
        overlay = orig_img.copy()
        overlay[(predict == 1) & (mask2d == 1)] = [0, 255, 0]
        overlay[(predict == 1) & (mask2d == 0)] = [0, 0, 255]
        overlay[(predict == 0) & (mask2d == 1)] = [255, 0, 0]
        Image.fromarray(overlay).save(os.path.join(vis_path, "mix_overlay", f"{name}_dice{dice:.2f}.png"))

    return dice, iou


def test(save_path, log):
    """
    Run model evaluation on the test set with optional visualization.
    Returns average Dice and IoU.
    """
    # Directories
    weight_path = os.path.join(save_path, f"models/best_model-{config.model_name}.pth.tar")
    vis_path = os.path.join(save_path, "visualize_test")
    for subdir in ["predict_binary", "predict_heatmap", "mix_overlay"]:
        os.makedirs(os.path.join(vis_path, subdir), exist_ok=True)

    # Model
    model = get_model(config.model_name, config).cuda()
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Load weights
    checkpoint = torch.load(weight_path, map_location="cuda")
    load_res = model.load_state_dict(checkpoint["state_dict"], strict=False)
    log.info(f"Missing keys: {load_res.missing_keys}")
    log.info(f"Unexpected keys: {load_res.unexpected_keys}")
    log.info("Model loaded successfully!\n")

    # Dataset
    test_loader = DataLoader(get_dataset("Test_Folder", augment=False), batch_size=1, shuffle=False)
    # test_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    # test_text = read_text(os.path.join(config.dataset_root, 'Test_Folder/Test_text.xlsx'))
    # test_dataset = ImageToImage2D(os.path.join(config.dataset_root, 'Test_Folder'), config.task_name, test_text,
    #                               test_tf, image_size=config.img_size, token_len=config.token_len, mode="test")
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluation
    dice_sum, iou_sum, total_time = 0.0, 0.0, 0.0
    with tqdm(total=len(test_loader), desc="Test visualize", unit="img", ncols=120) as pbar:
        for i, (sample, names) in enumerate(test_loader, 1):
            image, mask, text_token, text_mask = (
                sample["image"], sample["mask"], sample["text_token"], sample["text_mask"]
            )
            start_time = time.time()
            dice, iou = vis_and_save(model, image, text_token, text_mask, mask, vis_path, names)
            elapsed = time.time() - start_time

            dice_sum += dice
            iou_sum += iou
            total_time += elapsed
            pbar.set_postfix({"Dice": round(dice_sum / i, 4),
                              "IoU": round(iou_sum / i, 4),
                              "Time": f"{(total_time / i) * 1000:.2f} ms/img"})
            pbar.update()

    avg_dice = dice_sum / len(test_loader)
    avg_iou = iou_sum / len(test_loader)
    avg_time = total_time / len(test_loader)

    return avg_dice, avg_iou, avg_time

