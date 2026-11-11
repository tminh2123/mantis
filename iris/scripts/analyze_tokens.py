#!/usr/bin/env python3
"""
Analyze token distribution in QaTa dataset.

Statistics:
- Token count per text (words and ClinicalBERT tokens)
- Distribution histogram
- Max token length check vs model's max_token_len

Usage:
    python scripts/analyze_tokens.py --data_dir ../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/train
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset folder with texts.json')
    parser.add_argument('--max_token_len', type=int, default=18, help='Model max token length')
    args = parser.parse_args()
    
    # Load texts
    texts_path = Path(args.data_dir) / 'texts.json'
    with open(texts_path) as f:
        texts = json.load(f)
    
    print(f"Dataset: {args.data_dir}")
    print(f"Total samples: {len(texts)}")
    print(f"Model max_token_len: {args.max_token_len}")
    
    # Word-level stats (simple split)
    word_counts = [len(t.split()) for t in texts.values()]
    
    print(f"\n=== Word-level Statistics ===")
    print(f"Min words: {min(word_counts)}")
    print(f"Max words: {max(word_counts)}")
    print(f"Mean words: {np.mean(word_counts):.1f}")
    print(f"Median words: {np.median(word_counts):.1f}")
    
    # Distribution
    word_dist = Counter(word_counts)
    print(f"\nWord count distribution:")
    for count in sorted(word_dist.keys()):
        pct = word_dist[count] / len(texts) * 100
        bar = '#' * int(pct / 2)
        print(f"  {count:2d} words: {word_dist[count]:5d} ({pct:5.1f}%) {bar}")
    
    # ClinicalBERT tokenization
    print(f"\n=== ClinicalBERT Token Statistics ===")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        token_counts = []
        truncated = []
        
        for text in texts.values():
            tokens = tokenizer.tokenize(text)
            token_counts.append(len(tokens))
            if len(tokens) + 2 > args.max_token_len:  # +2 for [CLS] and [SEP]
                truncated.append((text, len(tokens)))
        
        print(f"Min tokens: {min(token_counts)}")
        print(f"Max tokens: {max(token_counts)}")
        print(f"Mean tokens: {np.mean(token_counts):.1f}")
        print(f"Median tokens: {np.median(token_counts):.1f}")
        
        # Check truncation
        effective_max = args.max_token_len - 2  # Remove [CLS] and [SEP]
        n_truncated = sum(1 for c in token_counts if c > effective_max)
        print(f"\nTruncation analysis (max_token_len={args.max_token_len}, effective={effective_max}):")
        print(f"  Samples truncated: {n_truncated} ({n_truncated/len(texts)*100:.1f}%)")
        
        if truncated:
            print(f"\nSample truncated texts (showing first 5):")
            for text, count in sorted(truncated, key=lambda x: -x[1])[:5]:
                print(f"  [{count} tokens] '{text[:80]}...'")
        
        # Token distribution
        token_dist = Counter(token_counts)
        print(f"\nToken count distribution:")
        for count in sorted(token_dist.keys()):
            pct = token_dist[count] / len(texts) * 100
            bar = '#' * int(pct / 2)
            truncate_mark = ' [TRUNCATED]' if count > effective_max else ''
            print(f"  {count:2d} tokens: {token_dist[count]:5d} ({pct:5.1f}%) {bar}{truncate_mark}")
            
    except ImportError:
        print("transformers not installed, skipping ClinicalBERT analysis")
    
    # Unique texts
    unique_texts = set(texts.values())
    print(f"\n=== Unique Text Analysis ===")
    print(f"Unique texts: {len(unique_texts)}")
    print(f"Duplication rate: {(1 - len(unique_texts)/len(texts))*100:.1f}%")


if __name__ == '__main__':
    main()
