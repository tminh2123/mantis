#!/usr/bin/env python3
"""
Analyze prediction errors by text attributes:
- one/two/three infected areas
- unilateral/bilateral
- left/right
- upper/middle/lower

Output: output/analysis/prediction_errors/
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_attributes(text: str) -> dict:
    """Extract attributes from text description."""
    text_lower = text.lower()
    
    # Laterality
    if 'bilateral' in text_lower:
        laterality = 'bilateral'
    elif 'unilateral' in text_lower:
        laterality = 'unilateral'
    else:
        laterality = 'unknown'
    
    # Number of infected areas
    if 'one infected' in text_lower:
        num_areas = 'one'
    elif 'two infected' in text_lower:
        num_areas = 'two'
    elif 'three infected' in text_lower:
        num_areas = 'three'
    elif 'four infected' in text_lower:
        num_areas = 'four'
    else:
        num_areas = 'unknown'
    
    # Side (left/right)
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
    
    # Vertical position
    has_upper = 'upper' in text_lower
    has_middle = 'middle' in text_lower
    has_lower = 'lower' in text_lower
    
    positions = []
    if has_upper: positions.append('upper')
    if has_middle: positions.append('middle')
    if has_lower: positions.append('lower')
    
    if len(positions) == 0:
        vertical = 'none'
    elif len(positions) == 1:
        vertical = positions[0]
    else:
        vertical = 'multi'
    
    return {
        'laterality': laterality,
        'num_areas': num_areas,
        'side': side,
        'vertical': vertical,
    }


def main():
    # Load dice results
    dice_path = PROJECT_ROOT / 'output/analysis/dice_distribution/dice_results.json'
    with open(dice_path) as f:
        dice_results = json.load(f)
    
    print(f"Total test samples: {len(dice_results)}")
    
    # Extract attributes for all samples
    for r in dice_results:
        attrs = extract_attributes(r['text'])
        r.update(attrs)
    
    # Analyze by each attribute
    output_dir = PROJECT_ROOT / 'output/analysis/prediction_errors'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    attributes = ['laterality', 'num_areas', 'side', 'vertical']
    
    print("\n" + "=" * 70)
    print("PREDICTION ERROR ANALYSIS BY TEXT ATTRIBUTES")
    print("=" * 70)
    
    summary = {}
    
    for attr in attributes:
        print(f"\n{'='*70}")
        print(f"BY {attr.upper()}")
        print(f"{'='*70}")
        
        # Group by attribute value
        groups = defaultdict(list)
        for r in dice_results:
            groups[r[attr]].append(r)
        
        attr_summary = {}
        
        for value in sorted(groups.keys()):
            samples = groups[value]
            n = len(samples)
            
            # Calculate stats
            dices = [s['dice'] for s in samples]
            mean_dice = sum(dices) / len(dices)
            
            # Count by dice ranges
            n_excellent = sum(1 for d in dices if d >= 0.85)
            n_good = sum(1 for d in dices if 0.70 <= d < 0.85)
            n_poor = sum(1 for d in dices if 0.50 <= d < 0.70)
            n_worst = sum(1 for d in dices if d < 0.50)
            
            print(f"\n{value:12s}: {n:4d} samples, mean Dice = {mean_dice*100:.1f}%")
            print(f"  Excellent (≥85%): {n_excellent:4d} ({n_excellent/n*100:5.1f}%)")
            print(f"  Good (70-85%):    {n_good:4d} ({n_good/n*100:5.1f}%)")
            print(f"  Poor (50-70%):    {n_poor:4d} ({n_poor/n*100:5.1f}%)")
            print(f"  Worst (<50%):     {n_worst:4d} ({n_worst/n*100:5.1f}%)")
            
            attr_summary[value] = {
                'count': n,
                'mean_dice': mean_dice,
                'excellent': n_excellent,
                'good': n_good,
                'poor': n_poor,
                'worst': n_worst,
                'worst_pct': n_worst / n * 100,
            }
        
        summary[attr] = attr_summary
    
    # Cross analysis: laterality x num_areas
    print(f"\n{'='*70}")
    print("CROSS ANALYSIS: LATERALITY x NUM_AREAS")
    print(f"{'='*70}")
    
    cross = defaultdict(list)
    for r in dice_results:
        key = f"{r['laterality']}_{r['num_areas']}"
        cross[key].append(r)
    
    print(f"\n{'Category':<25s} {'N':>6s} {'Mean':>8s} {'<50%':>8s} {'%Worst':>8s}")
    print("-" * 60)
    
    cross_summary = {}
    for key in sorted(cross.keys()):
        samples = cross[key]
        n = len(samples)
        dices = [s['dice'] for s in samples]
        mean_dice = sum(dices) / len(dices)
        n_worst = sum(1 for d in dices if d < 0.50)
        
        print(f"{key:<25s} {n:>6d} {mean_dice*100:>7.1f}% {n_worst:>8d} {n_worst/n*100:>7.1f}%")
        
        cross_summary[key] = {
            'count': n,
            'mean_dice': mean_dice,
            'worst': n_worst,
            'worst_pct': n_worst / n * 100,
        }
    
    summary['cross_laterality_num_areas'] = cross_summary
    
    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(dice_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
