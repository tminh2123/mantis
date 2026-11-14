#!/usr/bin/env python3
"""
Analyze what model predictions are confused about:
- left/right confusion
- one/two area confusion (predicts bilateral when text says unilateral)
- upper/middle/lower confusion

Compares PREDICTION vs GT (not text vs GT).

Output: output/analysis/prediction_confusion/
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

import torch
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import IRISDataset
from models.iris import IRIS


def get_mask_side(mask: np.ndarray, threshold: float = 0.7) -> str:
    """Determine which side mask is on."""
    h, w = mask.shape
    mid_x = w // 2
    
    left_sum = mask[:, :mid_x].sum()
    right_sum = mask[:, mid_x:].sum()
    total = left_sum + right_sum
    
    if total == 0:
        return 'empty'
    
    left_ratio = left_sum / total
    if left_ratio > threshold:
        return 'left'
    elif left_ratio < (1 - threshold):
        return 'right'
    else:
        return 'both'


def get_mask_vertical(mask: np.ndarray, threshold: float = 0.6) -> str:
    """Determine vertical position of mask."""
    h, w = mask.shape
    third = h // 3
    
    upper_sum = mask[:third, :].sum()
    middle_sum = mask[third:2*third, :].sum()
    lower_sum = mask[2*third:, :].sum()
    total = upper_sum + middle_sum + lower_sum
    
    if total == 0:
        return 'empty'
    
    upper_r = upper_sum / total
    middle_r = middle_sum / total
    lower_r = lower_sum / total
    
    if upper_r > threshold:
        return 'upper'
    elif middle_r > threshold:
        return 'middle'
    elif lower_r > threshold:
        return 'lower'
    else:
        return 'multi'


def count_regions(mask: np.ndarray, min_area: int = 100) -> int:
    """Count number of separate regions in mask."""
    import cv2
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    
    # Filter by area (exclude background label 0)
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count


def extract_text_attrs(text: str) -> dict:
    """Extract attributes from text."""
    text_lower = text.lower()
    
    # Laterality
    if 'bilateral' in text_lower:
        laterality = 'bilateral'
    elif 'unilateral' in text_lower:
        laterality = 'unilateral'
    else:
        laterality = 'unknown'
    
    # Side
    has_left = 'left' in text_lower
    has_right = 'right' in text_lower
    if has_left and has_right:
        side = 'both'
    elif has_left:
        side = 'left'
    elif has_right:
        side = 'right'
    else:
        side = 'none'
    
    return {'text_laterality': laterality, 'text_side': side}


def main():
    # Paths
    config_path = PROJECT_ROOT / 'configs/qata.yaml'
    checkpoint_path = PROJECT_ROOT / 'output/checkpoints/qata_fusion_h8_e4/best.pth'
    dice_results_path = PROJECT_ROOT / 'output/analysis/dice_distribution/dice_results.json'
    output_dir = PROJECT_ROOT / 'output/analysis/prediction_confusion'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    test_dir = (PROJECT_ROOT / cfg['data']['test_dir']).resolve()
    dataset = IRISDataset(str(test_dir), cfg['data']['image_size'], augment=False)
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
    
    # Load dice results for filtering
    with open(dice_results_path) as f:
        dice_results = json.load(f)
    
    # Filter worst samples
    worst = [r for r in dice_results if r['dice'] < 0.5]
    worst_idx = {r['idx'] for r in worst}
    print(f"Analyzing {len(worst)} worst samples (Dice < 50%)")
    
    # Analyze confusion
    results = []
    
    confusion_counts = {
        'side': {'correct': 0, 'confused': 0, 'na': 0},
        'vertical': {'correct': 0, 'confused': 0, 'na': 0},
        'num_regions': {'correct': 0, 'fewer': 0, 'more': 0, 'na': 0},
    }
    
    confused_samples = {
        'side': [],
        'vertical': [],
        'num_regions': [],
    }
    
    for i, r in enumerate(worst):
        idx = r['idx']
        sample = dataset[idx]
        img = sample['image']
        mask_gt = sample['mask'].squeeze().numpy()
        text = sample['text']
        
        # Get prediction
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            pred = model(img_tensor, [text])
            if isinstance(pred, dict):
                pred = pred['pred']
            mask_pred = (torch.sigmoid(pred) > 0.5).float().cpu()[0, 0].numpy()
        
        # Analyze GT
        gt_side = get_mask_side(mask_gt)
        gt_vert = get_mask_vertical(mask_gt)
        gt_regions = count_regions(mask_gt)
        
        # Analyze Prediction
        pred_side = get_mask_side(mask_pred)
        pred_vert = get_mask_vertical(mask_pred)
        pred_regions = count_regions(mask_pred)
        
        # Text attributes
        text_attrs = extract_text_attrs(text)
        
        result = {
            'idx': idx,
            'image_id': r['image_id'],
            'dice': r['dice'],
            'text': text,
            'gt_side': gt_side,
            'pred_side': pred_side,
            'gt_vert': gt_vert,
            'pred_vert': pred_vert,
            'gt_regions': gt_regions,
            'pred_regions': pred_regions,
            **text_attrs,
        }
        results.append(result)
        
        # Count side confusion
        if gt_side in ['left', 'right'] and pred_side in ['left', 'right']:
            if gt_side == pred_side:
                confusion_counts['side']['correct'] += 1
            else:
                confusion_counts['side']['confused'] += 1
                confused_samples['side'].append(result)
        else:
            confusion_counts['side']['na'] += 1
        
        # Count vertical confusion
        if gt_vert in ['upper', 'middle', 'lower'] and pred_vert in ['upper', 'middle', 'lower']:
            if gt_vert == pred_vert:
                confusion_counts['vertical']['correct'] += 1
            else:
                confusion_counts['vertical']['confused'] += 1
                confused_samples['vertical'].append(result)
        else:
            confusion_counts['vertical']['na'] += 1
        
        # Count region confusion
        if gt_regions > 0:
            if pred_regions == gt_regions:
                confusion_counts['num_regions']['correct'] += 1
            elif pred_regions < gt_regions:
                confusion_counts['num_regions']['fewer'] += 1
                confused_samples['num_regions'].append(result)
            else:
                confusion_counts['num_regions']['more'] += 1
                confused_samples['num_regions'].append(result)
        else:
            confusion_counts['num_regions']['na'] += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREDICTION CONFUSION ANALYSIS (Worst samples, Dice < 50%)")
    print("=" * 70)
    
    print(f"\n--- SIDE (Left/Right) Confusion ---")
    s = confusion_counts['side']
    total_s = s['correct'] + s['confused']
    if total_s > 0:
        print(f"  Correct:  {s['correct']:3d} ({s['correct']/total_s*100:.1f}%)")
        print(f"  Confused: {s['confused']:3d} ({s['confused']/total_s*100:.1f}%)")
    print(f"  N/A:      {s['na']:3d}")
    
    print(f"\n--- VERTICAL (Upper/Middle/Lower) Confusion ---")
    v = confusion_counts['vertical']
    total_v = v['correct'] + v['confused']
    if total_v > 0:
        print(f"  Correct:  {v['correct']:3d} ({v['correct']/total_v*100:.1f}%)")
        print(f"  Confused: {v['confused']:3d} ({v['confused']/total_v*100:.1f}%)")
    print(f"  N/A:      {v['na']:3d}")
    
    print(f"\n--- NUM REGIONS Confusion ---")
    n = confusion_counts['num_regions']
    total_n = n['correct'] + n['fewer'] + n['more']
    if total_n > 0:
        print(f"  Correct:       {n['correct']:3d} ({n['correct']/total_n*100:.1f}%)")
        print(f"  Predicts fewer:{n['fewer']:3d} ({n['fewer']/total_n*100:.1f}%)")
        print(f"  Predicts more: {n['more']:3d} ({n['more']/total_n*100:.1f}%)")
    print(f"  N/A:           {n['na']:3d}")
    
    # Detailed confused samples
    print(f"\n--- Side Confused Samples ({len(confused_samples['side'])}) ---")
    for r in confused_samples['side'][:10]:
        print(f"  [{r['dice']*100:5.1f}%] GT={r['gt_side']:5s} Pred={r['pred_side']:5s} | {r['image_id'][:40]}")
    
    print(f"\n--- Vertical Confused Samples ({len(confused_samples['vertical'])}) ---")
    for r in confused_samples['vertical'][:10]:
        print(f"  [{r['dice']*100:5.1f}%] GT={r['gt_vert']:6s} Pred={r['pred_vert']:6s} | {r['image_id'][:40]}")
    
    print(f"\n--- Region Confused Samples ({len(confused_samples['num_regions'])}) ---")
    for r in confused_samples['num_regions'][:10]:
        print(f"  [{r['dice']*100:5.1f}%] GT={r['gt_regions']} regions, Pred={r['pred_regions']} | {r['image_id'][:40]}")
    
    # Save
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({
            'total_worst': len(worst),
            'confusion_counts': confusion_counts,
        }, f, indent=2)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    for key in confused_samples:
        with open(output_dir / f'{key}_confused.json', 'w') as f:
            json.dump(confused_samples[key], f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
