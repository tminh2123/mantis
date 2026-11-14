#!/usr/bin/env python3
"""Visualize worst samples (Dice < 50%) - static output, no interaction."""

import sys
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import IRISDataset


def main():
    # Paths
    config_path = PROJECT_ROOT / 'configs/qata.yaml'
    dice_results_path = PROJECT_ROOT / 'output/analysis/dice_distribution/dice_results.json'
    output_dir = PROJECT_ROOT / 'output/analysis/worst_samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Resolve test dir
    test_dir = (PROJECT_ROOT / cfg['data']['test_dir']).resolve()
    print(f"Test dir: {test_dir}")
    
    # Load dataset (no model needed - just show GT)
    dataset = IRISDataset(
        root_dir=str(test_dir),
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    print(f"Dataset: {len(dataset)} samples")
    
    # Load dice results
    with open(dice_results_path) as f:
        dice_results = json.load(f)
    
    # Filter worst samples (< 50%)
    worst = [r for r in dice_results if r['dice'] < 0.5]
    worst = sorted(worst, key=lambda x: x['dice'])
    print(f"Samples with Dice < 50%: {len(worst)}")
    
    # Create figure with all worst samples
    n_samples = len(worst)
    if n_samples == 0:
        print("No samples with Dice < 50%!")
        return
    
    # Plot each sample: image + GT mask overlay
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, r in enumerate(worst):
        idx = r['idx']
        dice = r['dice']
        text = r['text']
        
        sample = dataset[idx]
        img = sample['image'].permute(1, 2, 0).numpy()
        mask = sample['mask'].squeeze().numpy()
        
        # Left: original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"[{dice*100:.1f}%] {r['image_id']}", fontsize=9)
        axes[i, 0].axis('off')
        
        # Right: image + GT overlay
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask, cmap='Reds', alpha=0.5)
        axes[i, 1].set_title(f"Text: {text[:60]}...", fontsize=8)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'worst_samples_under_50pct.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    # Also save individual images for easier viewing
    print("\nSaving individual images...")
    for i, r in enumerate(worst):
        idx = r['idx']
        dice = r['dice']
        
        sample = dataset[idx]
        img = sample['image'].permute(1, 2, 0).numpy()
        mask = sample['mask'].squeeze().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img)
        ax.imshow(mask, cmap='Reds', alpha=0.5)
        ax.set_title(f"Dice: {dice*100:.1f}%\n{r['text']}", fontsize=9)
        ax.axis('off')
        
        individual_path = output_dir / f"{i+1:02d}_dice{dice*100:.0f}_{r['image_id']}.png"
        plt.savefig(individual_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  {individual_path.name}")
    
    print(f"\nDone! Check {output_dir}")


if __name__ == '__main__':
    main()
