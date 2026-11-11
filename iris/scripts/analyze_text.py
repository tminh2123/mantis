"""
IRIS - Text Pattern Analysis
Parse templated texts and count component frequencies.

Template format:
"{Bilateral/Unilateral} pulmonary infection, {N} infected area(s), {locations}."
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_text(text: str) -> dict:
    """Parse a templated text into components."""
    result = {
        'laterality': None,      # Bilateral / Unilateral
        'num_areas': None,       # 1, 2, 3, ...
        'locations': [],         # List of location strings
        'left_locations': [],    # Locations on left
        'right_locations': [],   # Locations on right
    }
    
    text_lower = text.lower()
    
    # Laterality
    if 'bilateral' in text_lower:
        result['laterality'] = 'Bilateral'
    elif 'unilateral' in text_lower:
        result['laterality'] = 'Unilateral'
    
    # Number of areas
    num_match = re.search(r'(\w+) infected area', text_lower)
    if num_match:
        num_str = num_match.group(1)
        num_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
        result['num_areas'] = num_map.get(num_str, num_str)
    
    # Parse locations - everything after "area(s),"
    loc_match = re.search(r'infected areas?,\s*(.+?)\.?$', text, re.IGNORECASE)
    if loc_match:
        loc_str = loc_match.group(1).strip().rstrip('.')
        
        # Split by " and " to separate left/right parts
        parts = re.split(r'\s+and\s+', loc_str)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Determine if left or right
            if 'left' in part.lower():
                result['left_locations'].append(part)
            elif 'right' in part.lower():
                result['right_locations'].append(part)
            
            result['locations'].append(part)
    
    return result


def analyze_patterns(data_dir: str, output_dir: str):
    """Analyze text patterns in dataset."""
    texts_json = os.path.join(data_dir, 'texts.json')
    
    with open(texts_json, 'r') as f:
        texts_data = json.load(f)
    
    print(f"Total samples: {len(texts_data)}")
    
    # Get unique texts
    unique_texts = list(set(texts_data.values()))
    print(f"Unique texts: {len(unique_texts)}")
    
    # Count frequencies
    text_counts = Counter(texts_data.values())
    
    # Parse all texts
    parsed = [parse_text(t) for t in unique_texts]
    
    # Aggregate statistics
    laterality_counts = Counter(p['laterality'] for p in parsed)
    num_areas_counts = Counter(p['num_areas'] for p in parsed)
    
    # Location analysis
    all_left_locs = []
    all_right_locs = []
    for p in parsed:
        all_left_locs.extend(p['left_locations'])
        all_right_locs.extend(p['right_locations'])
    
    left_loc_counts = Counter(all_left_locs)
    right_loc_counts = Counter(all_right_locs)
    
    # Position keywords
    position_keywords = ['upper', 'middle', 'lower', 'all']
    left_positions = Counter()
    right_positions = Counter()
    
    for loc in all_left_locs:
        loc_lower = loc.lower()
        for pos in position_keywords:
            if pos in loc_lower:
                left_positions[pos] += 1
    
    for loc in all_right_locs:
        loc_lower = loc.lower()
        for pos in position_keywords:
            if pos in loc_lower:
                right_positions[pos] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("TEXT PATTERN ANALYSIS")
    print("=" * 60)
    
    print("\n--- LATERALITY ---")
    for lat, count in laterality_counts.most_common():
        print(f"  {lat}: {count} unique texts")
    
    print("\n--- NUMBER OF AREAS ---")
    for num, count in num_areas_counts.most_common():
        print(f"  {num} area(s): {count} unique texts")
    
    print("\n--- LEFT LUNG LOCATIONS ---")
    for loc, count in left_loc_counts.most_common(15):
        print(f"  [{count:3d}] {loc}")
    
    print("\n--- RIGHT LUNG LOCATIONS ---")
    for loc, count in right_loc_counts.most_common(15):
        print(f"  [{count:3d}] {loc}")
    
    print("\n--- LEFT POSITION KEYWORDS ---")
    for pos, count in left_positions.most_common():
        print(f"  {pos}: {count}")
    
    print("\n--- RIGHT POSITION KEYWORDS ---")
    for pos, count in right_positions.most_common():
        print(f"  {pos}: {count}")
    
    # Most/least common texts
    print("\n--- TOP 10 MOST COMMON TEXTS ---")
    for text, count in text_counts.most_common(10):
        text_short = text[:60] + "..." if len(text) > 60 else text
        print(f"  [{count:4d}] {text_short}")
    
    print("\n--- TOP 10 LEAST COMMON TEXTS ---")
    for text, count in text_counts.most_common()[-10:]:
        text_short = text[:60] + "..." if len(text) > 60 else text
        print(f"  [{count:4d}] {text_short}")
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Laterality pie
    axes[0, 0].pie(laterality_counts.values(), labels=laterality_counts.keys(), 
                   autopct='%1.1f%%', colors=['steelblue', 'coral'])
    axes[0, 0].set_title('Laterality Distribution')
    
    # Number of areas bar
    nums = list(num_areas_counts.keys())
    counts = list(num_areas_counts.values())
    axes[0, 1].bar(range(len(nums)), counts, tick_label=[str(n) for n in nums])
    axes[0, 1].set_xlabel('Number of Infected Areas')
    axes[0, 1].set_ylabel('Count (unique texts)')
    axes[0, 1].set_title('Number of Areas Distribution')
    
    # Text frequency histogram
    freq_values = list(text_counts.values())
    axes[0, 2].hist(freq_values, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Sample Count per Text')
    axes[0, 2].set_ylabel('Number of Texts')
    axes[0, 2].set_title(f'Text Frequency Distribution (n={len(unique_texts)})')
    axes[0, 2].axvline(np.mean(freq_values), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(freq_values):.1f}')
    axes[0, 2].legend()
    
    # Left positions
    left_pos_names = [p for p, _ in left_positions.most_common()]
    left_pos_counts = [c for _, c in left_positions.most_common()]
    axes[1, 0].bar(range(len(left_pos_names)), left_pos_counts, tick_label=left_pos_names, color='steelblue')
    axes[1, 0].set_title('Left Lung Position Keywords')
    axes[1, 0].set_ylabel('Count')
    
    # Right positions
    right_pos_names = [p for p, _ in right_positions.most_common()]
    right_pos_counts = [c for _, c in right_positions.most_common()]
    axes[1, 1].bar(range(len(right_pos_names)), right_pos_counts, tick_label=right_pos_names, color='coral')
    axes[1, 1].set_title('Right Lung Position Keywords')
    axes[1, 1].set_ylabel('Count')
    
    # Top 15 most frequent texts
    top_15 = text_counts.most_common(15)
    y_pos = np.arange(len(top_15))
    labels = [t[:35] + "..." if len(t) > 35 else t for t, _ in top_15]
    counts = [c for _, c in top_15]
    axes[1, 2].barh(y_pos, counts, color='lightgreen')
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(labels, fontsize=7)
    axes[1, 2].set_xlabel('Sample Count')
    axes[1, 2].set_title('Top 15 Most Frequent Texts')
    axes[1, 2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(output_dir, 'text_patterns.png')}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'text_patterns_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TEXT PATTERN SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples: {len(texts_data)}\n")
        f.write(f"Unique texts: {len(unique_texts)}\n")
        f.write(f"Avg samples per text: {len(texts_data)/len(unique_texts):.1f}\n\n")
        
        f.write("LATERALITY:\n")
        for lat, count in laterality_counts.most_common():
            f.write(f"  {lat}: {count}\n")
        
        f.write("\nNUMBER OF AREAS:\n")
        for num, count in num_areas_counts.most_common():
            f.write(f"  {num}: {count}\n")
        
        f.write("\nALL UNIQUE TEXTS:\n")
        for text, count in text_counts.most_common():
            f.write(f"[{count:4d}] {text}\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Text Pattern Analysis')
    parser.add_argument('--data_dir', required=True, help='Dataset directory')
    parser.add_argument('--output_dir', default='output/analysis/text_patterns', help='Output directory')
    args = parser.parse_args()
    
    analyze_patterns(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
