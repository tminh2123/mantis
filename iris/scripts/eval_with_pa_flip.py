"""
Evaluate IRIS with PA images flipped to AP orientation.

Hypothesis: PA view images are horizontally mirrored compared to AP.
By flipping PA images before inference and flipping predictions back,
model should perform better on PA cases.

Usage:
    python scripts/eval_with_pa_flip.py --config configs/qata.yaml --tag fusion_h8_e4
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import build_iris
from datasets import IRISDataset
from utils import load_config, get_device, compute_metrics_per_sample


def infer_config_from_checkpoint(ckpt_path: str, base_cfg: dict) -> dict:
    """Infer model config from checkpoint weights."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model']
    
    # Infer base_channel from out0.weight shape [1, base_channel, 1, 1]
    if 'out0.weight' in state:
        base_channel = state['out0.weight'].shape[1]
        base_cfg['model']['base_channel'] = base_channel
        print(f"Inferred base_channel: {base_channel}")
    
    # Check if model has attention (up0.att)
    has_att = any('up0.att' in k for k in state.keys())
    if has_att:
        base_cfg['model']['decoder_attention'] = True
        print("Inferred decoder_attention: True")
    
    return base_cfg, ckpt


def is_pa_view(filename: str) -> bool:
    """Check if image is PA view based on filename."""
    fname = filename.upper()
    # QaTa naming: contains _PA_ or ends with PA before extension
    return '_PA_' in fname or '_PA.' in fname


@torch.no_grad()
def evaluate_with_pa_flip(model, dataset, device, threshold=0.5):
    """Evaluate with PA images flipped to AP orientation."""
    model.eval()
    
    results = {
        'all': {'dice': [], 'iou': []},
        'ap': {'dice': [], 'iou': [], 'count': 0},
        'pa': {'dice': [], 'iou': [], 'count': 0},
        'pa_flipped': 0,
    }
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
        mask = sample['mask'].to(device)  # [1, H, W]
        text = sample['text']
        filepath = sample['path']
        filename = Path(filepath).name
        
        is_pa = is_pa_view(filename)
        
        # Flip PA images to AP orientation
        if is_pa:
            image = torch.flip(image, dims=[3])  # Horizontal flip
            results['pa_flipped'] += 1
        
        # Forward pass
        outputs = model(image, [text])
        pred = outputs['pred']  # [1, 1, H, W]
        
        # Flip prediction back for PA images
        if is_pa:
            pred = torch.flip(pred, dims=[3])
        
        # Compute metrics
        metrics = compute_metrics_per_sample(pred, mask.unsqueeze(0).unsqueeze(0), threshold)
        dice = metrics['dice'].item()
        iou = metrics['iou'].item()
        
        results['all']['dice'].append(dice)
        results['all']['iou'].append(iou)
        
        if is_pa:
            results['pa']['dice'].append(dice)
            results['pa']['iou'].append(iou)
            results['pa']['count'] += 1
        else:
            results['ap']['dice'].append(dice)
            results['ap']['iou'].append(iou)
            results['ap']['count'] += 1
    
    return results


@torch.no_grad()
def evaluate_baseline(model, dataset, device, threshold=0.5):
    """Evaluate without PA flip (baseline)."""
    model.eval()
    
    results = {
        'all': {'dice': [], 'iou': []},
        'ap': {'dice': [], 'iou': [], 'count': 0},
        'pa': {'dice': [], 'iou': [], 'count': 0},
    }
    
    for idx in tqdm(range(len(dataset)), desc="Baseline"):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask'].to(device)
        text = sample['text']
        filepath = sample['path']
        filename = Path(filepath).name
        
        is_pa = is_pa_view(filename)
        
        # No flip
        outputs = model(image, [text])
        pred = outputs['pred']
        
        metrics = compute_metrics_per_sample(pred, mask.unsqueeze(0).unsqueeze(0), threshold)
        dice = metrics['dice'].item()
        iou = metrics['iou'].item()
        
        results['all']['dice'].append(dice)
        results['all']['iou'].append(iou)
        
        if is_pa:
            results['pa']['dice'].append(dice)
            results['pa']['iou'].append(iou)
            results['pa']['count'] += 1
        else:
            results['ap']['dice'].append(dice)
            results['ap']['iou'].append(iou)
            results['ap']['count'] += 1
    
    return results


def print_results(results, title):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    all_dice = np.array(results['all']['dice']) * 100
    all_iou = np.array(results['all']['iou']) * 100
    
    print(f"\nOverall ({len(all_dice)} samples):")
    print(f"  Dice: {all_dice.mean():.2f}% ± {all_dice.std():.2f}%")
    print(f"  IoU:  {all_iou.mean():.2f}% ± {all_iou.std():.2f}%")
    
    # Worst rate (<50% Dice)
    worst = (all_dice < 50).sum()
    print(f"  Worst (<50%): {worst} ({worst/len(all_dice)*100:.1f}%)")
    
    # AP breakdown
    if results['ap']['count'] > 0:
        ap_dice = np.array(results['ap']['dice']) * 100
        ap_iou = np.array(results['ap']['iou']) * 100
        ap_worst = (ap_dice < 50).sum()
        print(f"\nAP view ({results['ap']['count']} samples):")
        print(f"  Dice: {ap_dice.mean():.2f}% ± {ap_dice.std():.2f}%")
        print(f"  IoU:  {ap_iou.mean():.2f}% ± {ap_iou.std():.2f}%")
        print(f"  Worst (<50%): {ap_worst} ({ap_worst/len(ap_dice)*100:.1f}%)")
    
    # PA breakdown
    if results['pa']['count'] > 0:
        pa_dice = np.array(results['pa']['dice']) * 100
        pa_iou = np.array(results['pa']['iou']) * 100
        pa_worst = (pa_dice < 50).sum()
        print(f"\nPA view ({results['pa']['count']} samples):")
        print(f"  Dice: {pa_dice.mean():.2f}% ± {pa_dice.std():.2f}%")
        print(f"  IoU:  {pa_iou.mean():.2f}% ± {pa_iou.std():.2f}%")
        print(f"  Worst (<50%): {pa_worst} ({pa_worst/len(pa_dice)*100:.1f}%)")
    
    if 'pa_flipped' in results:
        print(f"\nPA images flipped: {results['pa_flipped']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--tag', required=True, help='Experiment tag')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--base_channel', type=int, help='Base channel (override config)')
    parser.add_argument('--num_heads', type=int, help='Num heads (override config)')
    parser.add_argument('--expansion', type=int, help='Expansion (override config)')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Apply overrides
    if args.base_channel:
        cfg['model']['base_channel'] = args.base_channel
    if args.num_heads:
        cfg['model']['num_heads'] = args.num_heads
    if args.expansion:
        cfg['model']['expansion'] = args.expansion
    device = get_device()
    
    # Checkpoint path
    dataset = cfg['project']['dataset']
    exp_name = f"{dataset}_{args.tag}"
    ckpt_path = f"{args.output_dir}/checkpoints/{exp_name}/best.pth"
    
    if not Path(ckpt_path).exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return
    
    # Infer config from checkpoint and load weights
    cfg, ckpt = infer_config_from_checkpoint(ckpt_path, cfg)
    
    # Build model with inferred config
    model = build_iris(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint: {ckpt_path}")
    
    # Load test dataset
    test_dataset = IRISDataset(
        root_dir=cfg['data']['test_dir'],
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Count AP/PA
    ap_count = sum(1 for i in range(len(test_dataset)) 
                   if not is_pa_view(Path(test_dataset.samples[i][0]).name))
    pa_count = len(test_dataset) - ap_count
    print(f"AP: {ap_count}, PA: {pa_count}")
    
    # Baseline evaluation (no flip)
    print("\n" + "="*60)
    print("Running BASELINE (no PA flip)...")
    print("="*60)
    baseline_results = evaluate_baseline(model, test_dataset, device)
    print_results(baseline_results, "BASELINE (No PA Flip)")
    
    # Evaluation with PA flip
    print("\n" + "="*60)
    print("Running WITH PA FLIP...")
    print("="*60)
    flip_results = evaluate_with_pa_flip(model, test_dataset, device)
    print_results(flip_results, "WITH PA FLIP (PA -> AP orientation)")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    baseline_dice = np.mean(baseline_results['all']['dice']) * 100
    flip_dice = np.mean(flip_results['all']['dice']) * 100
    diff = flip_dice - baseline_dice
    
    print(f"\nOverall Dice:")
    print(f"  Baseline:  {baseline_dice:.2f}%")
    print(f"  With flip: {flip_dice:.2f}%")
    print(f"  Diff:      {diff:+.2f}%")
    
    if pa_count > 0:
        baseline_pa = np.mean(baseline_results['pa']['dice']) * 100
        flip_pa = np.mean(flip_results['pa']['dice']) * 100
        pa_diff = flip_pa - baseline_pa
        
        print(f"\nPA view Dice:")
        print(f"  Baseline:  {baseline_pa:.2f}%")
        print(f"  With flip: {flip_pa:.2f}%")
        print(f"  Diff:      {pa_diff:+.2f}%")


if __name__ == '__main__':
    main()
