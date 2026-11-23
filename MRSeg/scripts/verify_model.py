"""
Quick sanity check for MRSeg model.
Run from MRSeg/ directory:
    python scripts/verify_model.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.mrseg import build_mrseg

cfg = {
    "model": {
        "convnext_path": "pretrained_models/convnext-tiny-224",
        "bert_path":     "pretrained_models/pubmedbert",
        "max_token_len": 24,
        "lora_rank": 8, "lora_scale": 16.0, "lora_dropout": 0.05,
        "router_num_adapters": 4, "router_topk": 2,
        "pair_adapter_rank": 8, "pair_adapter_eta": 16.0,
        "num_regions_f3": 12, "num_regions_f4": 6,
        "decoder_dim": 192, "img_adapter_channels": 64,
    }
}

print("Building MRSegModel...")
model = build_mrseg(cfg)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable/1e6:.2f}M")
print(f"  Total:     {total/1e6:.2f}M")

B = 2
img   = torch.zeros(B, 1, 224, 224)
texts = ["bilateral pulmonary infection"] * B

print("\nRunning forward pass (B=2, 1ch, 224x224)...")
model.eval()
with torch.no_grad():
    out = model(img, texts)

pred = out["pred"]
print(f"  Output shape: {pred.shape}")
assert pred.shape == (B, 1, 224, 224), f"Unexpected shape: {pred.shape}"
print("\nOK — MRSeg verified successfully.")
