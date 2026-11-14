#!/usr/bin/env python3
"""Visualize worst samples (Dice < 50%) with Image, GT, and Prediction."""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import IRISDataset
from models.iris import IRIS


def main():
    # Paths
    config_path = PROJECT_ROOT / 'configs/qata.yaml'
    checkpoint_path = PROJECT_ROOT / 'output/checkpoints/qata_fusion_h8_e4/best.pth'
    dice_results_path = PROJECT_ROOT / 'output/analysis/dice_distribution/dice_results.json'
    output_dir = PROJECT_ROOT / 'output/analysis/worst_samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Resolve test dir
    test_dir = (PROJECT_ROOT / cfg['data']['test_dir']).resolve()
    print(f"Test dir: {test_dir}")
    
    # Load dataset
    dataset = IRISDataset(
        root_dir=str(test_dir),
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    print(f"Dataset: {len(dataset)} samples")
    
    # Load model
    print("Loading model...")
    model = IRIS(
        visual_encoder='resnet',
        text_encoder='clinicalbert',
        fusion='bifmd',
        num_classes=1,
        img_size=cfg['data']['image_size'],
        base_channel=64,
        num_heads=8,
        expansion=4,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print("Model loaded!")
    
    # Load dice results
    with open(dice_results_path) as f:
        dice_results = json.load(f)
    
    # Filter worst samples (< 50%)
    worst = [r for r in dice_results if r['dice'] < 0.5]
    worst = sorted(worst, key=lambda x: x['dice'])
    print(f"Samples with Dice < 50%: {len(worst)}")
    
    if len(worst) == 0:
        print("No samples with Dice < 50%!")
        return
    
    # Save individual images: 3 columns (Image, GT, Prediction)
    print("\nGenerating visualizations...")
    for i, r in enumerate(worst):
        idx = r['idx']
        dice = r['dice']
        text = r['text']
        
        sample = dataset[idx]
        img = sample['image']
        mask = sample['mask'].squeeze().numpy()
        
        # Get prediction
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            pred = model(img_tensor, [text])
            if isinstance(pred, dict):
                pred = pred['pred']
            pred_mask = (torch.sigmoid(pred) > 0.5).float().cpu()[0, 0].numpy()
        
        # Convert image for display
        img_np = img.permute(1, 2, 0).numpy()
        
        # Create figure: 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Column 1: Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image', fontsize=10)
        axes[0].axis('off')
        
        # Column 2: Ground Truth (red overlay)
        axes[1].imshow(img_np)
        axes[1].imshow(mask, cmap='Reds', alpha=0.5)
        axes[1].set_title(f'Ground Truth (area: {mask.sum():.0f}px)', fontsize=10)
        axes[1].axis('off')
        
        # Column 3: Prediction (blue overlay)
        axes[2].imshow(img_np)
        axes[2].imshow(pred_mask, cmap='Blues', alpha=0.5)
        axes[2].set_title(f'Prediction (area: {pred_mask.sum():.0f}px)', fontsize=10)
        axes[2].axis('off')
        
        # Main title with dice and text
        fig.suptitle(f"Dice: {dice*100:.1f}% | {text}", fontsize=9, y=1.02)
        plt.tight_layout()
        
        # Save
        filename = f"{i+1:02d}_dice{dice*100:.0f}_{r['image_id']}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [{i+1}/{len(worst)}] {filename}")
    
    print(f"\nDone! Output: {output_dir}")


if __name__ == '__main__':
    main()
