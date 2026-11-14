#!/usr/bin/env python3
"""
Text ablation study: How important is text for segmentation?

Experiments:
1. Correct text (baseline)
2. Random text (shuffle between samples)
3. Fixed text ("pulmonary infection")
4. Empty text ("")

Usage:
    python scripts/ablation_text.py \
        --config configs/qata.yaml \
        --checkpoint output/checkpoints/qata_fusion_h8_e4/best.pth \
        --visual resnet --text clinicalbert --fusion bifmd \
        --num_heads 8 --expansion 4
"""

import argparse
import os
import sys
import random
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
    dice = (2 * intersection / (union + 1e-8)).item()
    return dice


def evaluate_with_texts(model, dataset, text_fn, device, desc="Evaluating"):
    """
    Evaluate model with custom text function.
    
    Args:
        text_fn: Function(idx, original_text, all_texts) -> text to use
    """
    model.eval()
    all_texts = [dataset[i]['text'] for i in range(len(dataset))]
    
    dice_scores = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=desc):
            sample = dataset[idx]
            img = sample['image'].unsqueeze(0).to(device)
            mask = sample['mask'].to(device)
            
            # Ensure mask is [1, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            elif mask.dim() == 3 and mask.shape[0] != 1:
                mask = mask.unsqueeze(0)
            
            original_text = sample['text']
            
            # Get text to use
            text = text_fn(idx, original_text, all_texts)
            
            # Forward
            pred = model(img, [text])
            if isinstance(pred, dict):
                pred = pred['pred']
            
            dice = compute_dice(pred, mask)
            dice_scores.append(dice)
            
            # Debug first sample
            if idx == 0:
                pred_sig = torch.sigmoid(pred)
                print(f"\n[DEBUG] First sample:")
                print(f"  Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
                print(f"  Mask shape: {mask.shape}, sum: {mask.sum().item():.0f}")
                print(f"  Pred shape: {pred.shape}, range: [{pred.min():.3f}, {pred.max():.3f}]")
                print(f"  Pred sigmoid range: [{pred_sig.min():.3f}, {pred_sig.max():.3f}]")
                print(f"  Pred > 0.5 sum: {(pred_sig > 0.5).sum().item()}")
                print(f"  Dice: {dice:.4f}")
    
    return dice_scores


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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    print(f"Loaded: {args.checkpoint}")
    
    # Load test dataset
    dataset = IRISDataset(
        root_dir=cfg['data']['test_dir'],
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    print(f"Test samples: {len(dataset)}")
    
    # Prepare shuffled indices for random text
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)
    
    # Define text functions
    text_fns = {
        'correct': lambda idx, orig, all_t: orig,
        'random': lambda idx, orig, all_t: all_t[shuffled_indices[idx]],
        'fixed': lambda idx, orig, all_t: "pulmonary infection",
        'empty': lambda idx, orig, all_t: "",
    }
    
    # Run ablations
    print("\n" + "="*60)
    print("TEXT ABLATION STUDY")
    print("="*60)
    
    results = {}
    for name, text_fn in text_fns.items():
        scores = evaluate_with_texts(model, dataset, text_fn, device, f"[{name}]")
        mean_dice = np.mean(scores) * 100
        std_dice = np.std(scores) * 100
        results[name] = (mean_dice, std_dice)
        print(f"\n{name:>10}: Dice = {mean_dice:.2f}% ± {std_dice:.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    baseline = results['correct'][0]
    for name, (mean, std) in results.items():
        diff = mean - baseline
        print(f"{name:>10}: {mean:.2f}% (Δ = {diff:+.2f}%)")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    random_drop = baseline - results['random'][0]
    fixed_drop = baseline - results['fixed'][0]
    empty_drop = baseline - results['empty'][0]
    
    if random_drop < 1.0:
        print("⚠️  Random text ≈ Correct text")
        print("   → Text provides LITTLE discriminative information")
        print("   → Model relies mostly on visual features")
    else:
        print(f"✓  Random text drops {random_drop:.1f}%")
        print("   → Text IS important for segmentation")
    
    if fixed_drop < 2.0:
        print(f"⚠️  Fixed text ≈ Correct text (drop {fixed_drop:.1f}%)")
        print("   → Specific text content doesn't matter much")
    
    if empty_drop > 5.0:
        print(f"✓  Empty text drops {empty_drop:.1f}%")
        print("   → Model needs SOME text signal")


if __name__ == '__main__':
    main()
