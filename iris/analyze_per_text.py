"""
IRIS - Per-Text Dice Analysis
Group test samples by text, compute avg model Dice for each text group.
"""
import argparse
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import build_iris
from datasets import get_dataloader
from utils import load_config, get_device, load_checkpoint


def analyze_per_text(
    model, 
    dataloader, 
    device, 
    output_dir: str,
    threshold: float = 0.5
):
    """
    Group samples by text, compute avg model Dice for each group.
    """
    model.eval()
    
    # Collect results per text
    text_results = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            texts = batch['text']
            
            # Forward
            outputs = model(images, texts)
            preds = outputs['pred'] if isinstance(outputs, dict) else outputs
            preds = torch.sigmoid(preds)
            
            for i in range(len(images)):
                pred = preds[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()
                text = texts[i]
                
                # Compute Dice
                pred_bin = (pred > threshold).astype(np.float32)
                intersection = (pred_bin * mask).sum()
                dice = (2 * intersection + 1e-6) / (pred_bin.sum() + mask.sum() + 1e-6) * 100
                
                text_results[text].append(dice)
    
    # Aggregate per text
    results = []
    for text, dices in text_results.items():
        results.append({
            'text': text,
            'count': len(dices),
            'mean_dice': np.mean(dices),
            'std_dice': np.std(dices),
            'min_dice': np.min(dices),
            'max_dice': np.max(dices),
        })
    
    # Sort by mean_dice
    results.sort(key=lambda x: x['mean_dice'])
    
    # Print statistics
    print("\n" + "=" * 80)
    print("PER-TEXT DICE ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal unique texts: {len(results)}")
    print(f"Total samples: {sum(r['count'] for r in results)}")
    
    all_mean_dices = [r['mean_dice'] for r in results]
    all_counts = [r['count'] for r in results]
    
    print(f"\nOverall stats:")
    print(f"  Mean of text means: {np.mean(all_mean_dices):.2f}%")
    print(f"  Std of text means: {np.std(all_mean_dices):.2f}%")
    print(f"  Min text mean: {np.min(all_mean_dices):.2f}%")
    print(f"  Max text mean: {np.max(all_mean_dices):.2f}%")
    
    # Worst 10 texts
    print("\n" + "-" * 80)
    print("WORST 10 TEXTS (lowest Dice):")
    print("-" * 80)
    for r in results[:10]:
        text_short = r['text'][:50] + "..." if len(r['text']) > 50 else r['text']
        print(f"  [{r['count']:3d}] Dice: {r['mean_dice']:5.1f}% ± {r['std_dice']:4.1f}%  | {text_short}")
    
    # Best 10 texts
    print("\n" + "-" * 80)
    print("BEST 10 TEXTS (highest Dice):")
    print("-" * 80)
    for r in results[-10:]:
        text_short = r['text'][:50] + "..." if len(r['text']) > 50 else r['text']
        print(f"  [{r['count']:3d}] Dice: {r['mean_dice']:5.1f}% ± {r['std_dice']:4.1f}%  | {text_short}")
    
    # Correlation: count vs dice
    correlation = np.corrcoef(all_counts, all_mean_dices)[0, 1]
    print(f"\nCorrelation (count vs dice): {correlation:.3f}")
    
    # Create plots
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Histogram of mean dice per text
    axes[0, 0].hist(all_mean_dices, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_mean_dices), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(all_mean_dices):.1f}%')
    axes[0, 0].set_xlabel('Mean Dice per Text (%)')
    axes[0, 0].set_ylabel('Number of Texts')
    axes[0, 0].set_title('Distribution of Per-Text Dice')
    axes[0, 0].legend()
    
    # Count vs Dice scatter
    axes[0, 1].scatter(all_counts, all_mean_dices, alpha=0.5)
    axes[0, 1].set_xlabel('Sample Count per Text')
    axes[0, 1].set_ylabel('Mean Dice (%)')
    axes[0, 1].set_title(f'Count vs Dice (r={correlation:.3f})')
    
    # Worst 20 texts bar
    worst_20 = results[:20]
    y_pos = np.arange(len(worst_20))
    axes[1, 0].barh(y_pos, [r['mean_dice'] for r in worst_20], 
                     xerr=[r['std_dice'] for r in worst_20], color='salmon')
    axes[1, 0].set_yticks(y_pos)
    labels = [r['text'][:30] + "..." if len(r['text']) > 30 else r['text'] for r in worst_20]
    axes[1, 0].set_yticklabels(labels, fontsize=7)
    axes[1, 0].set_xlabel('Mean Dice (%)')
    axes[1, 0].set_title('WORST 20 Texts')
    axes[1, 0].invert_yaxis()
    
    # Best 20 texts bar
    best_20 = results[-20:]
    y_pos = np.arange(len(best_20))
    axes[1, 1].barh(y_pos, [r['mean_dice'] for r in best_20],
                     xerr=[r['std_dice'] for r in best_20], color='lightgreen')
    axes[1, 1].set_yticks(y_pos)
    labels = [r['text'][:30] + "..." if len(r['text']) > 30 else r['text'] for r in best_20]
    axes[1, 1].set_yticklabels(labels, fontsize=7)
    axes[1, 1].set_xlabel('Mean Dice (%)')
    axes[1, 1].set_title('BEST 20 Texts')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_text_dice.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(output_dir, 'per_text_dice.png')}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'per_text_dice.csv')
    with open(csv_path, 'w') as f:
        f.write("count,mean_dice,std_dice,min_dice,max_dice,text\n")
        for r in results:
            text_escaped = r['text'].replace('"', '""')
            f.write(f"{r['count']},{r['mean_dice']:.2f},{r['std_dice']:.2f},{r['min_dice']:.2f},{r['max_dice']:.2f},\"{text_escaped}\"\n")
    
    print(f"CSV saved to: {csv_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Per-Text Dice Analysis')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--output_dir', default='output/per_text_analysis', help='Output directory')
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
        batch_size=32,
        image_size=cfg['data']['image_size'],
        num_workers=cfg['data']['num_workers'],
        augment=False,
        shuffle=False
    )
    
    # Analyze
    exp_name = Path(args.checkpoint).parent.name
    output_dir = os.path.join(args.output_dir, exp_name)
    
    analyze_per_text(model, dataloader, device, output_dir)


if __name__ == '__main__':
    main()
