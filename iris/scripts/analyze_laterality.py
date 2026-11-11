"""
IRIS - Laterality Analysis Script
Analyze samples with "left"/"right" in text to check if model confuses sides.
"""
import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models import build_iris
from datasets import get_dataloader
from utils import load_config, get_device, load_checkpoint


def analyze_laterality(
    model, 
    dataloader, 
    device, 
    output_dir: str,
    threshold: float = 0.5
):
    """
    Analyze samples with laterality keywords (left/right).
    Check if model correctly segments the specified side.
    """
    model.eval()
    
    # Keywords to look for
    left_keywords = ['left', 'left lung', 'left lobe']
    right_keywords = ['right', 'right lung', 'right lobe']
    bilateral_keywords = ['bilateral', 'both']
    
    # Collect samples with laterality info
    left_samples = []
    right_samples = []
    bilateral_samples = []
    unilateral_samples = []  # Has left/right but NOT bilateral
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            texts = batch['text']
            paths = batch.get('path', [''] * len(images))
            
            # Forward
            outputs = model(images, texts)
            preds = outputs['pred'] if isinstance(outputs, dict) else outputs
            preds = torch.sigmoid(preds)
            
            for i in range(len(images)):
                text_lower = texts[i].lower()
                
                # Check keywords
                has_left = any(kw in text_lower for kw in left_keywords)
                has_right = any(kw in text_lower for kw in right_keywords)
                has_bilateral = any(kw in text_lower for kw in bilateral_keywords)
                
                if not (has_left or has_right or has_bilateral):
                    continue  # Skip samples without laterality info
                
                pred = preds[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                pred_bin = (pred > threshold).astype(np.float32)
                
                # Compute Dice
                intersection = (pred_bin * mask).sum()
                dice = (2 * intersection + 1e-6) / (pred_bin.sum() + mask.sum() + 1e-6) * 100
                
                # Compute left/right distribution
                h, w = mask.shape
                mid = w // 2
                
                gt_left = mask[:, :mid].sum()
                gt_right = mask[:, mid:].sum()
                gt_total = mask.sum() + 1e-6
                gt_left_ratio = gt_left / gt_total * 100
                gt_right_ratio = gt_right / gt_total * 100
                
                pred_left = pred_bin[:, :mid].sum()
                pred_right = pred_bin[:, mid:].sum()
                pred_total = pred_bin.sum() + 1e-6
                pred_left_ratio = pred_left / pred_total * 100
                pred_right_ratio = pred_right / pred_total * 100
                
                sample = {
                    'image': img,
                    'mask': mask,
                    'pred': pred,
                    'pred_bin': pred_bin,
                    'text': texts[i],
                    'dice': dice,
                    'path': paths[i],
                    'gt_left_ratio': gt_left_ratio,
                    'gt_right_ratio': gt_right_ratio,
                    'pred_left_ratio': pred_left_ratio,
                    'pred_right_ratio': pred_right_ratio,
                    'has_left': has_left,
                    'has_right': has_right,
                    'has_bilateral': has_bilateral,
                }
                
                if has_bilateral:
                    bilateral_samples.append(sample)
                elif has_left and not has_right:
                    left_samples.append(sample)
                    unilateral_samples.append(sample)
                elif has_right and not has_left:
                    right_samples.append(sample)
                    unilateral_samples.append(sample)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("LATERALITY ANALYSIS")
    print("=" * 60)
    
    print(f"\nSample counts:")
    print(f"  'left' only: {len(left_samples)}")
    print(f"  'right' only: {len(right_samples)}")
    print(f"  'bilateral': {len(bilateral_samples)}")
    print(f"  Unilateral (left OR right, not bilateral): {len(unilateral_samples)}")
    
    # Analyze unilateral samples - are predictions on correct side?
    if unilateral_samples:
        print("\n" + "-" * 40)
        print("UNILATERAL SAMPLES ANALYSIS")
        print("-" * 40)
        
        correct_side = 0
        wrong_side = 0
        
        for s in unilateral_samples:
            # For "left" samples, prediction should be mostly on left
            # For "right" samples, prediction should be mostly on right
            if s['has_left']:
                expected_dominant = 'left'
                actual_dominant = 'left' if s['pred_left_ratio'] > s['pred_right_ratio'] else 'right'
            else:
                expected_dominant = 'right'
                actual_dominant = 'left' if s['pred_left_ratio'] > s['pred_right_ratio'] else 'right'
            
            if expected_dominant == actual_dominant:
                correct_side += 1
            else:
                wrong_side += 1
        
        print(f"  Prediction on correct side: {correct_side}/{len(unilateral_samples)} ({100*correct_side/len(unilateral_samples):.1f}%)")
        print(f"  Prediction on WRONG side: {wrong_side}/{len(unilateral_samples)} ({100*wrong_side/len(unilateral_samples):.1f}%)")
        
        # Average Dice
        avg_dice = np.mean([s['dice'] for s in unilateral_samples])
        print(f"  Average Dice: {avg_dice:.2f}%")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize worst cases (wrong side or low dice)
    print("\n" + "-" * 40)
    print("VISUALIZING SAMPLES")
    print("-" * 40)
    
    # Sort by dice (worst first)
    all_laterality = left_samples + right_samples + bilateral_samples
    all_laterality.sort(key=lambda x: x['dice'])
    
    # Take worst N samples
    n_visualize = min(20, len(all_laterality))
    worst_samples = all_laterality[:n_visualize]
    
    # Create visualization
    for idx, sample in enumerate(worst_samples):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Determine label
        if sample['has_bilateral']:
            label = "Bilateral"
            color = "purple"
        elif sample['has_left']:
            label = "LEFT"
            color = "blue"
        else:
            label = "RIGHT"
            color = "green"
        
        # Image with center line
        axes[0].imshow(sample['image'])
        h, w, _ = sample['image'].shape
        axes[0].axvline(x=w//2, color='yellow', linestyle='--', linewidth=2, label='Center')
        axes[0].set_title(f"{label} | Dice: {sample['dice']:.1f}%", color=color, fontweight='bold', fontsize=14)
        axes[0].axis('off')
        
        # Ground truth with ratio
        axes[1].imshow(sample['mask'], cmap='gray')
        axes[1].axvline(x=w//2, color='yellow', linestyle='--', linewidth=2)
        axes[1].set_title(f"GT (L:{sample['gt_left_ratio']:.0f}% | R:{sample['gt_right_ratio']:.0f}%)", fontsize=12)
        axes[1].axis('off')
        
        # Prediction with ratio
        axes[2].imshow(sample['pred_bin'], cmap='gray')
        axes[2].axvline(x=w//2, color='yellow', linestyle='--', linewidth=2)
        axes[2].set_title(f"Pred (L:{sample['pred_left_ratio']:.0f}% | R:{sample['pred_right_ratio']:.0f}%)", fontsize=12)
        axes[2].axis('off')
        
        # Overlay
        overlay = sample['image'].copy()
        tp = (sample['pred_bin'] == 1) & (sample['mask'] == 1)
        fp = (sample['pred_bin'] == 1) & (sample['mask'] == 0)
        fn = (sample['pred_bin'] == 0) & (sample['mask'] == 1)
        overlay[tp] = [0, 1, 0]  # Green
        overlay[fp] = [1, 0, 0]  # Red
        overlay[fn] = [0, 0, 1]  # Blue
        axes[3].imshow(overlay)
        axes[3].axvline(x=w//2, color='yellow', linestyle='--', linewidth=2)
        axes[3].set_title("Overlay (G=TP, R=FP, B=FN)", fontsize=12)
        axes[3].axis('off')
        
        # Text as suptitle
        text = sample['text']
        if len(text) > 80:
            text = text[:77] + "..."
        plt.suptitle(f'Text: "{text}"', fontsize=11, y=1.02)
        
        plt.tight_layout()
        
        save_name = f"{idx:02d}_{label.lower()}_dice{sample['dice']:.0f}.png"
        plt.savefig(os.path.join(output_dir, save_name), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_name}")
    
    # Summary figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dice distribution by category
    if left_samples:
        axes[0, 0].hist([s['dice'] for s in left_samples], bins=20, alpha=0.7, label=f'Left (n={len(left_samples)})')
    if right_samples:
        axes[0, 0].hist([s['dice'] for s in right_samples], bins=20, alpha=0.7, label=f'Right (n={len(right_samples)})')
    if bilateral_samples:
        axes[0, 0].hist([s['dice'] for s in bilateral_samples], bins=20, alpha=0.7, label=f'Bilateral (n={len(bilateral_samples)})')
    axes[0, 0].set_xlabel('Dice Score (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Dice Distribution by Laterality')
    axes[0, 0].legend()
    
    # GT vs Pred left ratio for "left" samples
    if left_samples:
        gt_left = [s['gt_left_ratio'] for s in left_samples]
        pred_left = [s['pred_left_ratio'] for s in left_samples]
        axes[0, 1].scatter(gt_left, pred_left, alpha=0.5)
        axes[0, 1].plot([0, 100], [0, 100], 'r--', label='Perfect')
        axes[0, 1].set_xlabel('GT Left Ratio (%)')
        axes[0, 1].set_ylabel('Pred Left Ratio (%)')
        axes[0, 1].set_title('"LEFT" Samples: GT vs Pred Left Ratio')
        axes[0, 1].legend()
    
    # GT vs Pred right ratio for "right" samples
    if right_samples:
        gt_right = [s['gt_right_ratio'] for s in right_samples]
        pred_right = [s['pred_right_ratio'] for s in right_samples]
        axes[1, 0].scatter(gt_right, pred_right, alpha=0.5)
        axes[1, 0].plot([0, 100], [0, 100], 'r--', label='Perfect')
        axes[1, 0].set_xlabel('GT Right Ratio (%)')
        axes[1, 0].set_ylabel('Pred Right Ratio (%)')
        axes[1, 0].set_title('"RIGHT" Samples: GT vs Pred Right Ratio')
        axes[1, 0].legend()
    
    # Summary stats
    ax_text = axes[1, 1]
    ax_text.axis('off')
    
    stats_text = "SUMMARY STATISTICS\n\n"
    if left_samples:
        stats_text += f"LEFT samples (n={len(left_samples)}):\n"
        stats_text += f"  Avg Dice: {np.mean([s['dice'] for s in left_samples]):.1f}%\n"
        stats_text += f"  Avg GT Left Ratio: {np.mean([s['gt_left_ratio'] for s in left_samples]):.1f}%\n"
        stats_text += f"  Avg Pred Left Ratio: {np.mean([s['pred_left_ratio'] for s in left_samples]):.1f}%\n\n"
    
    if right_samples:
        stats_text += f"RIGHT samples (n={len(right_samples)}):\n"
        stats_text += f"  Avg Dice: {np.mean([s['dice'] for s in right_samples]):.1f}%\n"
        stats_text += f"  Avg GT Right Ratio: {np.mean([s['gt_right_ratio'] for s in right_samples]):.1f}%\n"
        stats_text += f"  Avg Pred Right Ratio: {np.mean([s['pred_right_ratio'] for s in right_samples]):.1f}%\n\n"
    
    if bilateral_samples:
        stats_text += f"BILATERAL samples (n={len(bilateral_samples)}):\n"
        stats_text += f"  Avg Dice: {np.mean([s['dice'] for s in bilateral_samples]):.1f}%\n"
    
    ax_text.text(0.1, 0.9, stats_text, transform=ax_text.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary saved to: {os.path.join(output_dir, 'summary.png')}")
    
    return {
        'left': left_samples,
        'right': right_samples,
        'bilateral': bilateral_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='IRIS Laterality Analysis')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--output_dir', default='output/analysis/laterality', help='Output directory')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    # Model overrides
    parser.add_argument('--visual', choices=['cnn', 'resnet'], help='Visual encoder')
    parser.add_argument('--text', choices=['clip', 'pubmedbert', 'clinicalbert'], help='Text encoder')
    parser.add_argument('--fusion', choices=['bcam', 'bifmd'], help='Fusion module')
    parser.add_argument('--num_heads', type=int, help='Attention heads')
    parser.add_argument('--expansion', type=int, help='MLP expansion')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Apply overrides
    if args.visual:
        cfg['model']['visual_encoder'] = args.visual
    if args.text:
        cfg['model']['text_encoder'] = args.text
    if args.fusion:
        cfg['model']['fusion'] = args.fusion
    if args.num_heads:
        cfg['model']['num_heads'] = args.num_heads
    if args.expansion:
        cfg['model']['expansion'] = args.expansion
    
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
        augment=False,
        shuffle=False
    )
    
    # Analyze
    exp_name = Path(args.checkpoint).parent.name
    output_dir = os.path.join(args.output_dir, exp_name)
    
    analyze_laterality(model, dataloader, device, output_dir)


if __name__ == '__main__':
    main()
