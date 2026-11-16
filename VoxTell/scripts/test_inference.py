"""Quick inference test for VoxTell.

Creates a synthetic NIfTI volume and runs inference with a text prompt.
Downloads model weights (~1.77GB) on first run.

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --input /path/to/real.nii.gz --prompt "liver"
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch


def make_synthetic_nifti(path: str, shape=(64, 64, 64)):
    """Create a small synthetic CT-like volume for testing."""
    rng = np.random.default_rng(42)
    # Simulate CT HU range roughly (-1000 to 400)
    vol = rng.normal(loc=-500, scale=200, size=shape).astype(np.float32)
    # Add a bright sphere to simulate an organ
    cx, cy, cz = [s // 2 for s in shape]
    r = shape[0] // 5
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    sphere = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2 < r**2
    vol[sphere] = 50.0

    affine = np.diag([1.5, 1.5, 1.5, 1])  # 1.5mm isotropic spacing
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, path)
    print(f"Saved synthetic volume: {shape} voxels @ {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default=None, help='Path to NIfTI file (optional)')
    parser.add_argument('--prompt', default='lung', help='Text prompt (default: lung)')
    parser.add_argument('--output', default='/tmp/voxtell_output', help='Output directory')
    args = parser.parse_args()

    from voxtell.inference.predictor import VoxTellPredictor

    # Prepare input
    if args.input:
        input_path = args.input
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        input_path = tmp.name
        tmp.close()
        make_synthetic_nifti(input_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading VoxTell predictor (downloads weights on first run ~1.77GB)...")

    t0 = time.time()
    predictor = VoxTellPredictor(device=device)
    print(f"Loaded in {time.time()-t0:.1f}s")

    print(f"Running inference: prompt='{args.prompt}'")
    t1 = time.time()
    predictor.predict_from_files(
        list_of_lists_or_source_folder=[[input_path]],
        output_folder_or_list_of_truncated_output_files=str(output_dir),
        list_of_text_prompts=[[args.prompt]],
        overwrite=True,
    )
    print(f"Inference done in {time.time()-t1:.1f}s")
    print(f"Output saved to: {output_dir}/")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM: {peak:.2f} GB")


if __name__ == '__main__':
    main()
