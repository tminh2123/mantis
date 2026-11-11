"""
IRIS - Intra-text Dice Analysis
Compute Dice between masks that share the same text description.
"""
import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


def load_mask(mask_path: str) -> np.ndarray:
    """Load and binarize mask."""
    mask = np.array(Image.open(mask_path).convert('L'))
    return (mask > 127).astype(np.float32)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice between two masks."""
    intersection = (mask1 * mask2).sum()
    return (2 * intersection + 1e-6) / (mask1.sum() + mask2.sum() + 1e-6) * 100


def analyze_intra_text_dice(data_dir: str, output_dir: str):
    """
    Group samples by text, compute pairwise Dice within each group.
    """
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    text_dir = os.path.join(data_dir, 'texts')
    
    # Group by text
    text_to_samples = defaultdict(list)
    
    for txt_file in os.listdir(text_dir):
        if not txt_file.endswith('.txt'):
            continue
        
        sample_id = txt_file.replace('.txt', '')
        txt_path = os.path.join(text_dir, txt_file)
        
        with open(txt_path, 'r') as f:
            text = f.read().strip()
        
        # Find corresponding mask
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(mask_dir, sample_id + ext)
            if os.path.exists(p):
                mask_path = p
                break
        
        if mask_path:
            text_to_samples[text].append({
                'id': sample_id,
                'mask_path': mask_path,
                'text': text
            })
    
    print(f"Found {len(text_to_samples)} unique texts")
    print(f"Total samples: {sum(len(v) for v in text_to_samples.values())}")
    
    # Compute intra-text Dice for each group
    results = []
    
    for text, samples in tqdm(text_to_samples.items(), desc="Computing intra-text Dice"):
        n = len(samples)
        
        if n < 2:
            # Can't compute pairwise Dice with single sample
            results.append({
                'text': text,
                'count': n,
                'mean_dice': None,
                'std_dice': None,
                'min_dice': None,
                'max_dice': None,
                'pairwise_dices': []
            })
            continue
        
        # Load all masks
        masks = []
        for s in samples:
            masks.append(load_mask(s['mask_path']))
        
        # Compute pairwise Dice
        pairwise_dices = []
        for i in range(n):
            for j in range(i + 1, n):
                dice = compute_dice(masks[i], masks[j])
                pairwise_dices.append(dice)
        
        results.append({
            'text': text,
            'count': n,
            'mean_dice': np.mean(pairwise_dices),
            'std_dice': np.std(pairwise_dices),
            'min_dice': np.min(pairwise_dices),
            'max_dice': np.max(pairwise_dices),
            'pairwise_dices': pairwise_dices
        })
    
    # Filter to groups with >= 2 samples
    valid_results = [r for r in results if r['mean_dice'] is not None]
    
    # Sort by count
    valid_results.sort(key=lambda x: x['count'], reverse=True)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("INTRA-TEXT DICE ANALYSIS")
    print("=" * 80)
    
    print(f"\nGroups with >= 2 samples: {len(valid_results)}")
    print(f"Groups with 1 sample (skipped): {len(results) - len(valid_results)}")
    
    # Overall statistics
    all_mean_dices = [r['mean_dice'] for r in valid_results]
    all_counts = [r['count'] for r in valid_results]
    
    print(f"\nOverall intra-text Dice:")
    print(f"  Mean of group means: {np.mean(all_mean_dices):.2f}%")
    print(f"  Std of group means: {np.std(all_mean_dices):.2f}%")
    print(f"  Min group mean: {np.min(all_mean_dices):.2f}%")
    print(f"  Max group mean: {np.max(all_mean_dices):.2f}%")
    
    # Top 10 most frequent texts
    print("\n" + "-" * 80)
    print("TOP 10 MOST FREQUENT TEXTS:")
    print("-" * 80)
    for r in valid_results[:10]:
        text_short = r['text'][:60] + "..." if len(r['text']) > 60 else r['text']
        print(f"  [{r['count']:3d} samples] Dice: {r['mean_dice']:5.1f}% ± {r['std_dice']:4.1f}%  | {text_short}")
    
    # Correlation: count vs mean_dice
    counts = np.array(all_counts)
    mean_dices = np.array(all_mean_dices)
    correlation = np.corrcoef(counts, mean_dices)[0, 1]
    
    print(f"\nCorrelation (count vs mean_dice): {correlation:.3f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Histogram of mean intra-text Dice
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    axes[0, 0].hist(all_mean_dices, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_mean_dices), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(all_mean_dices):.1f}%')
    axes[0, 0].set_xlabel('Mean Intra-text Dice (%)')
    axes[0, 0].set_ylabel('Number of Text Groups')
    axes[0, 0].set_title('Distribution of Intra-text Dice')
    axes[0, 0].legend()
    
    # Plot 2: Count vs Mean Dice scatter
    axes[0, 1].scatter(counts, mean_dices, alpha=0.5)
    axes[0, 1].set_xlabel('Sample Count per Text')
    axes[0, 1].set_ylabel('Mean Intra-text Dice (%)')
    axes[0, 1].set_title(f'Count vs Dice (r={correlation:.3f})')
    
    # Add trend line
    z = np.polyfit(counts, mean_dices, 1)
    p = np.poly1d(z)
    x_line = np.linspace(counts.min(), counts.max(), 100)
    axes[0, 1].plot(x_line, p(x_line), 'r--', alpha=0.7, label='Trend')
    axes[0, 1].legend()
    
    # Plot 3: Histogram of sample counts
    axes[1, 0].hist(all_counts, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Sample Count per Text')
    axes[1, 0].set_ylabel('Number of Text Groups')
    axes[1, 0].set_title('Distribution of Sample Counts')
    
    # Plot 4: Top 20 texts by count with their Dice
    top_20 = valid_results[:20]
    y_pos = np.arange(len(top_20))
    bars = axes[1, 1].barh(y_pos, [r['mean_dice'] for r in top_20], xerr=[r['std_dice'] for r in top_20])
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f"n={r['count']}" for r in top_20], fontsize=8)
    axes[1, 1].set_xlabel('Mean Intra-text Dice (%)')
    axes[1, 1].set_title('Top 20 Most Frequent Texts: Intra-text Dice')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intra_text_dice_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(output_dir, 'intra_text_dice_analysis.png')}")
    
    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'intra_text_dice_results.csv')
    with open(csv_path, 'w') as f:
        f.write("count,mean_dice,std_dice,min_dice,max_dice,text\n")
        for r in valid_results:
            text_escaped = r['text'].replace('"', '""')
            f.write(f"{r['count']},{r['mean_dice']:.2f},{r['std_dice']:.2f},{r['min_dice']:.2f},{r['max_dice']:.2f},\"{text_escaped}\"\n")
    
    print(f"CSV saved to: {csv_path}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    avg_dice = np.mean(all_mean_dices)
    if avg_dice < 50:
        print(f"\nLow intra-text Dice ({avg_dice:.1f}%): Masks with same text are VERY different.")
        print("→ Text provides minimal constraint on mask shape.")
        print("→ Visual features must do heavy lifting for segmentation.")
    elif avg_dice < 70:
        print(f"\nModerate intra-text Dice ({avg_dice:.1f}%): Masks with same text have some similarity.")
        print("→ Text provides coarse guidance (region).")
        print("→ Visual features refine exact boundaries.")
    else:
        print(f"\nHigh intra-text Dice ({avg_dice:.1f}%): Masks with same text are quite similar.")
        print("→ Text is highly predictive of mask.")
        print("→ Model can rely more on text-based priors.")
    
    if correlation < -0.3:
        print(f"\nNegative correlation ({correlation:.3f}): More samples → lower intra-Dice.")
        print("→ Frequent texts have more diverse masks (expected).")
    elif correlation > 0.3:
        print(f"\nPositive correlation ({correlation:.3f}): More samples → higher intra-Dice.")
        print("→ Unexpected! Frequent texts might be very specific.")
    else:
        print(f"\nWeak correlation ({correlation:.3f}): Count doesn't strongly predict Dice.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Intra-text Dice Analysis')
    parser.add_argument('--data_dir', required=True, help='Dataset directory (with images/, masks/, texts/)')
    parser.add_argument('--output_dir', default='output/intra_text_analysis', help='Output directory')
    args = parser.parse_args()
    
    analyze_intra_text_dice(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
