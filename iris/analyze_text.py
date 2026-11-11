"""
Analyze text patterns in QaTa-COV19 dataset.
"""
import json
from pathlib import Path
from collections import Counter
import re


def analyze_texts(texts_file: str):
    """Analyze text annotations."""
    with open(texts_file) as f:
        texts = json.load(f)
    
    print(f"Total samples: {len(texts)}")
    print()
    
    # Get all texts
    all_texts = list(texts.values())
    
    # Unique texts
    unique_texts = set(all_texts)
    print(f"Unique texts: {len(unique_texts)}")
    print()
    
    # Text frequency
    text_counts = Counter(all_texts)
    
    print("=" * 60)
    print("TEXT FREQUENCY (most common)")
    print("=" * 60)
    for text, count in text_counts.most_common(20):
        pct = count / len(all_texts) * 100
        print(f"[{count:4d}] ({pct:5.1f}%) {text[:80]}")
    
    print()
    print("=" * 60)
    print("TEXT LENGTH STATS")
    print("=" * 60)
    lengths = [len(t) for t in all_texts]
    word_counts = [len(t.split()) for t in all_texts]
    
    print(f"Char length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    print(f"Word count:  min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)/len(word_counts):.1f}")
    
    print()
    print("=" * 60)
    print("WORD FREQUENCY (top 30)")
    print("=" * 60)
    all_words = []
    for text in all_texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    for word, count in word_freq.most_common(30):
        print(f"[{count:5d}] {word}")
    
    print()
    print("=" * 60)
    print("ALL UNIQUE TEXTS")
    print("=" * 60)
    for i, text in enumerate(sorted(unique_texts), 1):
        count = text_counts[text]
        print(f"{i:3d}. [{count:4d}x] {text}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        texts_file = sys.argv[1]
    else:
        # Default QaTa path
        texts_file = "../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/train/texts.json"
    
    analyze_texts(texts_file)
