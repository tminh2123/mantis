"""
IRIS - Visualization Script
Visualize (image, text, mask prediction) triplets at different difficulty levels.
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models import build_iris
from datasets import get_dataloader
from utils import load_config, get_device, load_checkpoint, compute_metrics_per_sample


def visualize_samples(
    model, 
    dataloader, 
    device, 
    output_dir: str,
    num_samples: int = 12,
    threshold: float = 0.5
):
    """
    Visualize samples grouped by difficulty (Dice score).
    
    Groups:
    - Easy: Dice >= 90%
    - Medium: 70% <= Dice < 90%
    - Hard: Dice < 70%
    """
    model.eval()
    
    # Collect all predictions with metrics
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            texts = batch['text']
            paths = batch.get('path', [''] * len(images))
            
            # Forward
            outputs = model(images, texts)
            if isinstance(outputs, dict):
                preds = outputs['pred']
            else:
                preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            preds = torch.sigmoid(preds)
            
            # Process each sample
            for i in range(len(images)):
                pred = preds[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                
                # Denormalize image
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                # Compute Dice
                pred_bin = (pred > threshold).astype(np.float32)
                intersection = (pred_bin * mask).sum()
                dice = (2 * intersection + 1e-6) / (pred_bin.sum() + mask.sum() + 1e-6) * 100
                
                results.append({
                    'image': img,
                    'mask': mask,
                    'pred': pred,
                    'pred_bin': pred_bin,
                    'text': texts[i],
                    'dice': dice,
                    'path': paths[i] if isinstance(paths[i], str) else paths[i]
                })
    
    # Sort by Dice score
    results.sort(key=lambda x: x['dice'])
    
    # Group by difficulty
    easy = [r for r in results if r['dice'] >= 90]
    medium = [r for r in results if 70 <= r['dice'] < 90]
    hard = [r for r in results if r['dice'] < 70]
    
    print(f"\nSample distribution:")
    print(f"  Easy (Dice >= 90%): {len(easy)}")
    print(f"  Medium (70-90%): {len(medium)}")
    print(f"  Hard (< 70%): {len(hard)}")
    
    # Select samples for visualization
    n_per_group = num_samples // 3
    
    def select_samples(group, n):
        if len(group) <= n:
            return group
        # Select evenly spaced samples
        indices = np.linspace(0, len(group) - 1, n, dtype=int)
        return [group[i] for i in indices]
    
    selected = []
    selected.extend(select_samples(hard, n_per_group))  # Hard first
    selected.extend(select_samples(medium, n_per_group))
    selected.extend(select_samples(easy, n_per_group))
    
    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot grid
    n_cols = 4  # image, GT mask, pred mask, overlay
    n_rows = len(selected)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(selected):
        ax_img, ax_gt, ax_pred, ax_overlay = axes[idx]
        
        # Determine difficulty label
        if sample['dice'] >= 90:
            difficulty = "Easy"
            color = "green"
        elif sample['dice'] >= 70:
            difficulty = "Medium"
            color = "orange"
        else:
            difficulty = "Hard"
            color = "red"
        
        # Image
        ax_img.imshow(sample['image'])
        ax_img.set_title(f"{difficulty} | Dice: {sample['dice']:.1f}%", color=color, fontweight='bold')
        ax_img.axis('off')
        
        # Wrap text
        text = sample['text']
        if len(text) > 40:
            text = text[:37] + "..."
        ax_img.set_xlabel(f'"{text}"', fontsize=9)
        
        # Ground truth mask
        ax_gt.imshow(sample['mask'], cmap='gray')
        ax_gt.set_title("Ground Truth")
        ax_gt.axis('off')
        
        # Predicted mask
        ax_pred.imshow(sample['pred_bin'], cmap='gray')
        ax_pred.set_title("Prediction")
        ax_pred.axis('off')
        
        # Overlay
        overlay = sample['image'].copy()
        # Red for false positive, Green for true positive, Blue for false negative
        tp = (sample['pred_bin'] == 1) & (sample['mask'] == 1)
        fp = (sample['pred_bin'] == 1) & (sample['mask'] == 0)
        fn = (sample['pred_bin'] == 0) & (sample['mask'] == 1)
        
        overlay[tp] = [0, 1, 0]  # Green - correct
        overlay[fp] = [1, 0, 0]  # Red - false positive
        overlay[fn] = [0, 0, 1]  # Blue - false negative
        
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Overlay (G=TP, R=FP, B=FN)")
        ax_overlay.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {save_path}")
    
    # Also save individual samples
    individual_dir = os.path.join(output_dir, "samples")
    os.makedirs(individual_dir, exist_ok=True)
    
    for idx, sample in enumerate(selected):
        if sample['dice'] >= 90:
            prefix = f"{idx:02d}_easy"
        elif sample['dice'] >= 70:
            prefix = f"{idx:02d}_medium"
        else:
            prefix = f"{idx:02d}_hard"
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(sample['image'])
        axes[0].set_title(f"Dice: {sample['dice']:.1f}%")
        axes[0].axis('off')
        
        axes[1].imshow(sample['mask'], cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(sample['pred_bin'], cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        # Overlay
        overlay = sample['image'].copy()
        tp = (sample['pred_bin'] == 1) & (sample['mask'] == 1)
        fp = (sample['pred_bin'] == 1) & (sample['mask'] == 0)
        fn = (sample['pred_bin'] == 0) & (sample['mask'] == 1)
        overlay[tp] = [0, 1, 0]
        overlay[fp] = [1, 0, 0]
        overlay[fn] = [0, 0, 1]
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")
        axes[3].axis('off')
        
        plt.suptitle(f'Text: "{sample["text"]}"', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_dice{sample['dice']:.0f}.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Individual samples saved to: {individual_dir}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='IRIS Visualization')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--output_dir', default='output/visualizations', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=12, help='Number of samples to visualize')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    device = get_device()
    
    # Build model
    model = build_iris(cfg)
    model = model.to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Get dataloader
    split_dir = cfg['data'][f'{args.split}_dir']
    dataloader = get_dataloader(
        split_dir,
        batch_size=16,
        image_size=cfg['data']['image_size'],
        num_workers=cfg['data']['num_workers'],
        is_train=False,
        return_path=True
    )
    
    # Visualize
    exp_name = Path(args.checkpoint).parent.name
    output_dir = os.path.join(args.output_dir, exp_name)
    
    visualize_samples(
        model, 
        dataloader, 
        device, 
        output_dir,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
