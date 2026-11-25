"""Generate CSV files for DD-CMD from QaTa-COV19 dataset."""
import json
import os
from pathlib import Path
import pandas as pd

# Dataset paths (adjust for server)
DATASET_ROOT = "../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg"
OUTPUT_DIR = "data/qata"

def generate_csv(split: str):
    """Generate CSV for a dataset split."""
    split_dir = Path(DATASET_ROOT) / split
    
    # Load texts.json
    texts_path = split_dir / "texts.json"
    if texts_path.exists():
        with open(texts_path) as f:
            texts = json.load(f)
    else:
        # Default text if no texts.json
        texts = {}
        default_text = "Segment the COVID-19 infection region in this CT scan image."
    
    # Get all images
    img_dir = split_dir / "images"
    if not img_dir.exists():
        img_dir = split_dir / "img"
    
    rows = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        
        img_name = img_path.name
        # Get text - try different key formats
        text = texts.get(img_name) or texts.get(img_path.stem) or \
               texts.get(f"mask_{img_name}") or texts.get(f"mask_{img_path.stem}") or \
               "Segment the COVID-19 infection region in this CT scan image."
        
        rows.append({
            "Image": img_name,
            "text": text
        })
    
    # Save CSV
    df = pd.DataFrame(rows)
    output_path = Path(OUTPUT_DIR) / f"{split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"{split}: {len(df)} samples -> {output_path}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        generate_csv(split)
    print("Done!")
