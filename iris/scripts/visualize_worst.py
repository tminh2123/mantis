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
    
    # Analyze left-right confusion
    print("\n" + "="*60)
    print("LEFT-RIGHT CONFUSION ANALYSIS")
    print("="*60)
    
    n_confused = 0
    confused_samples = []
    
    for i, r in enumerate(worst):
        idx = r['idx']
        sample = dataset[idx]
        img = sample['image']
        mask = sample['mask'].squeeze().numpy()
        
        # Get prediction
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            pred = model(img_tensor, [sample['text']])
            if isinstance(pred, dict):
                pred = pred['pred']
            pred_mask = (torch.sigmoid(pred) > 0.5).float().cpu()[0, 0].numpy()
        
        # Calculate center of mass for GT and Prediction
        h, w = mask.shape
        mid_x = w // 2
        
        # GT: which side has more pixels?
        gt_left = mask[:, :mid_x].sum()
        gt_right = mask[:, mid_x:].sum()
        gt_total = gt_left + gt_right
        
        # Pred: which side has more pixels?
        pred_left = pred_mask[:, :mid_x].sum()
        pred_right = pred_mask[:, mid_x:].sum()
        pred_total = pred_left + pred_right
        
        if gt_total > 0 and pred_total > 0:
            gt_side = 'left' if gt_left > gt_right else 'right'
            pred_side = 'left' if pred_left > pred_right else 'right'
            
            # Check if confused (predict opposite side)
            gt_ratio = max(gt_left, gt_right) / gt_total
            pred_ratio = max(pred_left, pred_right) / pred_total
            
            # Only count if GT is clearly one-sided (>70%) and pred is opposite
            if gt_ratio > 0.7 and pred_ratio > 0.7 and gt_side != pred_side:
                n_confused += 1
                confused_samples.append({
                    'idx': idx,
                    'dice': r['dice'],
                    'text': r['text'],
                    'image_id': r['image_id'],
                    'gt_side': gt_side,
                    'pred_side': pred_side,
                    'gt_ratio': gt_ratio,
                    'pred_ratio': pred_ratio,
                })
    
    print(f"\nTotal samples < 50% Dice: {len(worst)}")
    print(f"Left-right confused: {n_confused} ({n_confused/len(worst)*100:.1f}%)")
    
    if confused_samples:
        print("\nConfused samples:")
        for s in confused_samples:
            print(f"  [{s['dice']*100:.1f}%] {s['image_id']}")
            print(f"    GT: {s['gt_side']} ({s['gt_ratio']*100:.0f}%), Pred: {s['pred_side']} ({s['pred_ratio']*100:.0f}%)")
            print(f"    Text: {s['text'][:60]}...")
    
    # Save analysis
    analysis_path = output_dir / 'left_right_analysis.json'
    analysis = {
        'total_worst': len(worst),
        'n_confused': n_confused,
        'confused_ratio': n_confused / len(worst) if worst else 0,
        'confused_samples': confused_samples,
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved: {analysis_path}")
    
    print(f"\nDone! Output: {output_dir}")


if __name__ == '__main__':
    main()
