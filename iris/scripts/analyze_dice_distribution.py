#!/usr/bin/env python3
"""
Analyze Dice score distribution on test set.

Output:
- Histogram of Dice scores
- Statistics (mean, median, percentiles)
- Breakdown by Dice ranges
- Worst samples analysis

Usage:
    python scripts/analyze_dice_distribution.py \
        --config configs/qata.yaml \
        --checkpoint output/checkpoints/qata_fusion_h8_e4/best.pth \
        --visual resnet --text clinicalbert --fusion bifmd \
        --num_heads 8 --expansion 4
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.iris import IRIS
from datasets.dataset import IRISDataset


def compute_dice(pred, target):
    """Compute Dice score."""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return (2 * intersection / (union + 1e-8)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--visual', type=str, default='resnet')
    parser.add_argument('--text', type=str, default='clinicalbert')
    parser.add_argument('--fusion', type=str, default='bifmd')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--memory_size', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output/analysis/dice_distribution')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Create model
    model = IRIS(
        visual_encoder=args.visual,
        text_encoder=args.text,
        fusion=args.fusion,
        num_classes=cfg.get('model', {}).get('num_classes', 1),
        img_size=cfg['data']['image_size'],
        base_channel=cfg.get('model', {}).get('base_channel', 64),
        num_heads=args.num_heads,
        expansion=args.expansion,
        memory_size=args.memory_size,
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt:
        state = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {args.checkpoint}")
    
    # Load test dataset
    dataset = IRISDataset(
        root_dir=cfg['data']['test_dir'],
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    print(f"Test samples: {len(dataset)}")
    
    # Compute Dice for all samples
    results = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Computing Dice"):
            sample = dataset[idx]
            img = sample['image'].unsqueeze(0).to(device)
            mask = sample['mask'].to(device)
            text = sample['text']
            path = sample['path']
            
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            
            pred = model(img, [text])
            if isinstance(pred, dict):
                pred = pred['pred']
            
            dice = compute_dice(pred, mask)
            
            results.append({
                'idx': idx,
                'image_id': os.path.basename(path),
                'text': text,
                'dice': dice,
                'mask_size': mask.sum().item(),
            })
    
    # Convert to numpy for analysis
    dice_scores = np.array([r['dice'] for r in results]) * 100
    
    # Print statistics
    print("\n" + "="*60)
    print("DICE SCORE DISTRIBUTION")
    print("="*60)
    
    print(f"\n=== Basic Statistics ===")
    print(f"  Samples: {len(dice_scores)}")
    print(f"  Mean: {np.mean(dice_scores):.2f}%")
    print(f"  Std: {np.std(dice_scores):.2f}%")
    print(f"  Median: {np.median(dice_scores):.2f}%")
    print(f"  Min: {np.min(dice_scores):.2f}%")
    print(f"  Max: {np.max(dice_scores):.2f}%")
    
    print(f"\n=== Percentiles ===")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th: {np.percentile(dice_scores, p):.2f}%")
    
    print(f"\n=== Distribution by Range ===")
    ranges = [
        (0, 10, "Failed (0-10%)"),
        (10, 30, "Very Poor (10-30%)"),
        (30, 50, "Poor (30-50%)"),
        (50, 70, "Below Avg (50-70%)"),
        (70, 85, "Average (70-85%)"),
        (85, 95, "Good (85-95%)"),
        (95, 100.1, "Excellent (95-100%)"),
    ]
    
    for low, high, label in ranges:
        count = np.sum((dice_scores >= low) & (dice_scores < high))
        pct = count / len(dice_scores) * 100
        bar = '#' * int(pct / 2)
        print(f"  {label:22s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Worst samples
    print(f"\n=== Worst 10 Samples ===")
    sorted_results = sorted(results, key=lambda x: x['dice'])
    for r in sorted_results[:10]:
        print(f"  [{r['dice']*100:5.1f}%] {r['image_id'][:40]}")
        print(f"         Text: '{r['text'][:60]}...'")
    
    # Best samples
    print(f"\n=== Best 10 Samples ===")
    for r in sorted_results[-10:]:
        print(f"  [{r['dice']*100:5.1f}%] {r['image_id'][:40]}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results as JSON
    output_file = os.path.join(args.output_dir, 'dice_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to: {output_file}")
    
    # Save summary
    summary = {
        'n_samples': len(dice_scores),
        'mean': float(np.mean(dice_scores)),
        'std': float(np.std(dice_scores)),
        'median': float(np.median(dice_scores)),
        'min': float(np.min(dice_scores)),
        'max': float(np.max(dice_scores)),
        'percentiles': {str(p): float(np.percentile(dice_scores, p)) for p in [5, 10, 25, 50, 75, 90, 95]},
        'distribution': {label: int(np.sum((dice_scores >= low) & (dice_scores < high))) 
                        for low, high, label in ranges},
    }
    
    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")


if __name__ == '__main__':
    main()
