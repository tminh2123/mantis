#!/usr/bin/env python3
"""
Analyze text-GT consistency: check if "left/right" in text matches GT mask position.

Output: output/analysis/text_gt_consistency/
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import IRISDataset


def get_text_side(text: str) -> str:
    """Extract left/right from text. Returns 'left', 'right', 'both', or 'none'."""
    text_lower = text.lower()
    has_left = 'left' in text_lower
    has_right = 'right' in text_lower
    
    if has_left and has_right:
        return 'both'
    elif has_left:
        return 'left'
    elif has_right:
        return 'right'
    else:
        return 'none'


def get_text_vertical(text: str) -> str:
    """Extract upper/middle/lower from text."""
    text_lower = text.lower()
    has_upper = 'upper' in text_lower
    has_middle = 'middle' in text_lower
    has_lower = 'lower' in text_lower
    
    positions = []
    if has_upper: positions.append('upper')
    if has_middle: positions.append('middle')
    if has_lower: positions.append('lower')
    
    if len(positions) == 0:
        return 'none'
    elif len(positions) == 1:
        return positions[0]
    else:
        return 'multi'


def get_gt_side(mask: np.ndarray, threshold: float = 0.7) -> str:
    """Determine which side GT mask is on. Returns 'left', 'right', 'both', or 'empty'."""
    h, w = mask.shape
    mid_x = w // 2
    
    left_sum = mask[:, :mid_x].sum()
    right_sum = mask[:, mid_x:].sum()
    total = left_sum + right_sum
    
    if total == 0:
        return 'empty'
    
    left_ratio = left_sum / total
    right_ratio = right_sum / total
    
    if left_ratio > threshold:
        return 'left'
    elif right_ratio > threshold:
        return 'right'
    else:
        return 'both'


def get_gt_vertical(mask: np.ndarray, threshold: float = 0.6) -> str:
    """Determine vertical position of GT mask. Returns 'upper', 'middle', 'lower', 'multi', or 'empty'."""
    h, w = mask.shape
    third = h // 3
    
    upper_sum = mask[:third, :].sum()
    middle_sum = mask[third:2*third, :].sum()
    lower_sum = mask[2*third:, :].sum()
    total = upper_sum + middle_sum + lower_sum
    
    if total == 0:
        return 'empty'
    
    upper_ratio = upper_sum / total
    middle_ratio = middle_sum / total
    lower_ratio = lower_sum / total
    
    # Check if concentrated in one region
    if upper_ratio > threshold:
        return 'upper'
    elif middle_ratio > threshold:
        return 'middle'
    elif lower_ratio > threshold:
        return 'lower'
    else:
        return 'multi'


def get_view_type(filename: str) -> str:
    """Extract view type from filename."""
    if 'vp-ap' in filename:
        return 'AP'
    elif 'vp-pa' in filename:
        return 'PA'
    elif 'vp-ll' in filename:
        return 'LL'
    else:
        return 'unknown'


def main():
    # Paths
    config_path = PROJECT_ROOT / 'configs/qata.yaml'
    output_dir = PROJECT_ROOT / 'output/analysis/text_gt_consistency'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Resolve test dir
    test_dir = (PROJECT_ROOT / cfg['data']['test_dir']).resolve()
    print(f"Test dir: {test_dir}")
    
    # Load dataset
    dataset = IRISDataset(
        root_dir=str(test_dir),
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    print(f"Dataset: {len(dataset)} samples\n")
    
    # Analyze each sample
    results = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        mask = sample['mask'].squeeze().numpy()
        text = sample['text']
        image_id = Path(sample['path']).name
        
        text_side = get_text_side(text)
        text_vert = get_text_vertical(text)
        gt_side = get_gt_side(mask)
        gt_vert = get_gt_vertical(mask)
        view_type = get_view_type(image_id)
        
        # Check horizontal consistency (only for unilateral cases)
        if text_side in ['left', 'right'] and gt_side in ['left', 'right']:
            h_consistent = (text_side == gt_side)
        else:
            h_consistent = None
        
        # Check vertical consistency
        if text_vert in ['upper', 'middle', 'lower'] and gt_vert in ['upper', 'middle', 'lower']:
            v_consistent = (text_vert == gt_vert)
        else:
            v_consistent = None
        
        results.append({
            'idx': idx,
            'image_id': image_id,
            'text': text,
            'text_side': text_side,
            'text_vert': text_vert,
            'gt_side': gt_side,
            'gt_vert': gt_vert,
            'view_type': view_type,
            'h_consistent': h_consistent,
            'v_consistent': v_consistent,
        })
    
    # Summary statistics
    print("=" * 60)
    print("TEXT-GT CONSISTENCY ANALYSIS")
    print("=" * 60)
    
    # Count text side distribution
    text_sides = Counter(r['text_side'] for r in results)
    print(f"\nText HORIZONTAL distribution:")
    for side, count in text_sides.most_common():
        print(f"  {side:6s}: {count:4d} ({count/len(results)*100:.1f}%)")
    
    # Count text vertical distribution
    text_verts = Counter(r['text_vert'] for r in results)
    print(f"\nText VERTICAL distribution:")
    for vert, count in text_verts.most_common():
        print(f"  {vert:6s}: {count:4d} ({count/len(results)*100:.1f}%)")
    
    # Count GT side distribution
    gt_sides = Counter(r['gt_side'] for r in results)
    print(f"\nGT HORIZONTAL distribution:")
    for side, count in gt_sides.most_common():
        print(f"  {side:6s}: {count:4d} ({count/len(results)*100:.1f}%)")
    
    # Count GT vertical distribution
    gt_verts = Counter(r['gt_vert'] for r in results)
    print(f"\nGT VERTICAL distribution:")
    for vert, count in gt_verts.most_common():
        print(f"  {vert:6s}: {count:4d} ({count/len(results)*100:.1f}%)")
    
    # Horizontal consistency analysis
    h_unilateral = [r for r in results if r['h_consistent'] is not None]
    h_consistent_count = sum(1 for r in h_unilateral if r['h_consistent'])
    h_inconsistent_count = len(h_unilateral) - h_consistent_count
    
    print(f"\n{'='*60}")
    print("HORIZONTAL (left/right) CONSISTENCY")
    print(f"{'='*60}")
    print(f"Unilateral samples: {len(h_unilateral)}")
    print(f"  Consistent: {h_consistent_count} ({h_consistent_count/len(h_unilateral)*100:.1f}%)")
    print(f"  Inconsistent: {h_inconsistent_count} ({h_inconsistent_count/len(h_unilateral)*100:.1f}%)")
    
    # Vertical consistency analysis
    v_single = [r for r in results if r['v_consistent'] is not None]
    v_consistent_count = sum(1 for r in v_single if r['v_consistent'])
    v_inconsistent_count = len(v_single) - v_consistent_count
    
    print(f"\n{'='*60}")
    print("VERTICAL (upper/middle/lower) CONSISTENCY")
    print(f"{'='*60}")
    print(f"Single-region samples: {len(v_single)}")
    print(f"  Consistent: {v_consistent_count} ({v_consistent_count/len(v_single)*100:.1f}%)")
    print(f"  Inconsistent: {v_inconsistent_count} ({v_inconsistent_count/len(v_single)*100:.1f}%)")
    
    # List inconsistent samples
    h_inconsistent = [r for r in h_unilateral if not r['h_consistent']]
    v_inconsistent = [r for r in v_single if not r['v_consistent']]
    
    print(f"\nHorizontal inconsistent samples ({len(h_inconsistent)}):")
    for r in h_inconsistent[:10]:
        print(f"  text={r['text_side']:5s} GT={r['gt_side']:5s} | {r['image_id'][:50]}")
    
    print(f"\nVertical inconsistent samples ({len(v_inconsistent)}):")
    for r in v_inconsistent[:10]:
        print(f"  text={r['text_vert']:6s} GT={r['gt_vert']:6s} | {r['image_id'][:50]}")
    if len(v_inconsistent) > 10:
        print(f"  ... and {len(v_inconsistent) - 10} more")
    
    # Save results
    summary = {
        'total_samples': len(results),
        'text_side_distribution': dict(text_sides),
        'text_vert_distribution': dict(text_verts),
        'gt_side_distribution': dict(gt_sides),
        'gt_vert_distribution': dict(gt_verts),
        'h_unilateral_samples': len(h_unilateral),
        'h_consistent_count': h_consistent_count,
        'h_inconsistent_count': h_inconsistent_count,
        'v_single_samples': len(v_single),
        'v_consistent_count': v_consistent_count,
        'v_inconsistent_count': v_inconsistent_count,
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / 'h_inconsistent_samples.json', 'w') as f:
        json.dump(h_inconsistent, f, indent=2)
    
    with open(output_dir / 'v_inconsistent_samples.json', 'w') as f:
        json.dump(v_inconsistent, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
