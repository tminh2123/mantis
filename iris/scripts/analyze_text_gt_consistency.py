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
        gt_side = get_gt_side(mask)
        view_type = get_view_type(image_id)
        
        # Check consistency (only for unilateral cases)
        if text_side in ['left', 'right'] and gt_side in ['left', 'right']:
            consistent = (text_side == gt_side)
        else:
            consistent = None  # Can't determine for bilateral/none
        
        results.append({
            'idx': idx,
            'image_id': image_id,
            'text': text,
            'text_side': text_side,
            'gt_side': gt_side,
            'view_type': view_type,
            'consistent': consistent,
        })
    
    # Summary statistics
    print("=" * 60)
    print("TEXT-GT CONSISTENCY ANALYSIS")
    print("=" * 60)
    
    # Count text side distribution
    text_sides = Counter(r['text_side'] for r in results)
    print(f"\nText side distribution:")
    for side, count in text_sides.most_common():
        print(f"  {side:6s}: {count:4d} ({count/len(results)*100:.1f}%)")
    
    # Count GT side distribution
    gt_sides = Counter(r['gt_side'] for r in results)
    print(f"\nGT side distribution:")
    for side, count in gt_sides.most_common():
        print(f"  {side:6s}: {count:4d} ({count/len(results)*100:.1f}%)")
    
    # Consistency analysis (only unilateral cases)
    unilateral = [r for r in results if r['consistent'] is not None]
    consistent_count = sum(1 for r in unilateral if r['consistent'])
    inconsistent_count = len(unilateral) - consistent_count
    
    print(f"\nUnilateral samples (text says left OR right, GT is left OR right):")
    print(f"  Total: {len(unilateral)}")
    print(f"  Consistent (text matches GT): {consistent_count} ({consistent_count/len(unilateral)*100:.1f}%)")
    print(f"  Inconsistent (text != GT): {inconsistent_count} ({inconsistent_count/len(unilateral)*100:.1f}%)")
    
    # Breakdown by view type
    print(f"\nInconsistent by view type:")
    inconsistent = [r for r in unilateral if not r['consistent']]
    view_counts = Counter(r['view_type'] for r in inconsistent)
    for view, count in view_counts.most_common():
        total_view = sum(1 for r in unilateral if r['view_type'] == view)
        print(f"  {view:3s}: {count:3d} / {total_view:3d} inconsistent ({count/total_view*100:.1f}%)")
    
    # List inconsistent samples
    print(f"\nInconsistent samples:")
    for r in inconsistent[:20]:  # First 20
        print(f"  [{r['view_type']:3s}] text={r['text_side']:5s} GT={r['gt_side']:5s} | {r['image_id'][:50]}")
    if len(inconsistent) > 20:
        print(f"  ... and {len(inconsistent) - 20} more")
    
    # Save results
    summary = {
        'total_samples': len(results),
        'text_side_distribution': dict(text_sides),
        'gt_side_distribution': dict(gt_sides),
        'unilateral_samples': len(unilateral),
        'consistent_count': consistent_count,
        'inconsistent_count': inconsistent_count,
        'inconsistent_by_view': dict(view_counts),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / 'inconsistent_samples.json', 'w') as f:
        json.dump(inconsistent, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
