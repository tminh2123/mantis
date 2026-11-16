"""Measure VRAM usage during a simulated training step.

No real data or pretrained weights needed — uses random tensors.

Usage:
    python scripts/measure_train_vram.py           # bs=2 (default)
    python scripts/measure_train_vram.py --bs 1
    python scripts/measure_train_vram.py --bs 4
"""

import argparse
import pydoc

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from batchgenerators.utilities.file_and_folder_operations import load_json


def gb(n_bytes):
    return f"{n_bytes / 1024**3:.2f} GB"


def snapshot(label):
    alloc = torch.cuda.memory_allocated()
    peak  = torch.cuda.max_memory_allocated()
    print(f"  [{label}] allocated={gb(alloc)}  peak={gb(peak)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=2, help='Batch size (default: 2)')
    parser.add_argument('--patch', type=int, default=192, help='Patch size (default: 192)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA available.")
        return

    from voxtell.model.voxtell_model import VoxTellModel

    # --- Build model from plans.json ---
    plans_path = hf_hub_download(repo_id='mrokuss/VoxTell', filename='voxtell_v1.1/plans.json')
    plans = load_json(plans_path)
    arch_kwargs = dict(**plans['configurations']['3d_fullres']['architecture']['arch_kwargs'])
    for key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

    torch.cuda.reset_peak_memory_stats()
    print(f"\nBatch size: {args.bs}  |  Patch: {args.patch}³")
    print("=" * 55)

    # --- Load model ---
    model = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    ).cuda().bfloat16().train()
    snapshot("model loaded")

    # --- Optimizer (AdamW fp32 states built on first step) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    snapshot("optimizer created (states not yet allocated)")

    # --- Synthetic inputs ---
    B, C, P = args.bs, 1, args.patch
    img   = torch.randn(B, C, P, P, P, device='cuda', dtype=torch.bfloat16)
    # text_embedding: (B, N_prompts, text_dim) — 1 prompt per sample
    text  = torch.randn(B, 1, 2560, device='cuda', dtype=torch.bfloat16)
    # fake target: same spatial shape as output (B, N, D, H, W)
    target = torch.zeros(B, 1, P, P, P, device='cuda', dtype=torch.bfloat16)
    snapshot("inputs created")

    # --- Forward ---
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(img, text)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
    snapshot("forward done")

    # --- Backward ---
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    snapshot("backward done  (peak = backward only)")

    # --- Optimizer step (allocates m, v states here) ---
    optimizer.step()
    optimizer.zero_grad()
    snapshot("optimizer step done")

    print("=" * 55)
    print(f"Peak VRAM total: {gb(torch.cuda.max_memory_allocated())}")


if __name__ == '__main__':
    main()
