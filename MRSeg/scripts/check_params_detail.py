"""Detailed trainable parameter breakdown for MRSeg."""
import sys
sys.path.insert(0, '.')

from models.mrseg import build_mrseg

cfg = {'model': {
    'convnext_path': 'pretrained_models/convnext-tiny-224',
    'bert_path': 'pretrained_models/pubmedbert'
}}

model = build_mrseg(cfg)

def count_params(module, name=""):
    """Count trainable params for a module."""
    total = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total

def count_named_params(model, prefix):
    """Count trainable params matching prefix in named_parameters."""
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n.startswith(prefix):
            total += p.numel()
    return total

print("=" * 60)
print("MRSeg Trainable Parameter Breakdown")
print("=" * 60)

# 1. Encoder LoRA
visual_lora = sum(p.numel() for n, p in model.named_parameters() 
                  if p.requires_grad and 'visual_enc' in n)
text_lora = sum(p.numel() for n, p in model.named_parameters() 
                if p.requires_grad and 'text_enc' in n)
print(f"\n[Encoder LoRA]")
print(f"  Visual (ConvNeXt) LoRA: {visual_lora:,}")
print(f"  Text (BERT) LoRA:       {text_lora:,}")
print(f"  Subtotal:               {visual_lora + text_lora:,}")

# 2. MedImageAdapter
med_adapter = count_params(model.med_adapter)
print(f"\n[MedImageAdapter]")
print(f"  Total: {med_adapter:,}")

# 3. Router
router = count_params(model.router)
print(f"\n[Router]")
print(f"  Total: {router:,}")
# Breakdown
mlp_w1 = 768 * 2 * 768  # (1536, 768)
mlp_b1 = 768
mlp_w2 = 768 * 4  # (768, 4)
mlp_b2 = 4
print(f"  - MLP L1: {mlp_w1 + mlp_b1:,} (1536→768)")
print(f"  - MLP L2: {mlp_w2 + mlp_b2:,} (768→4)")

# 4. PairAdapter
pair_adapter = count_params(model.pair_adapter)
print(f"\n[PairAdapter]")
print(f"  Total: {pair_adapter:,}")
# Breakdown
M, rank = 4, 8
f3_A = M * 384 * rank
f3_dw = M * rank * 3 * 3  # 3x3 depthwise
f3_B = M * rank * 384
f3_ln = 384 * 2  # weight + bias
f4_A = M * 768 * rank
f4_dw = M * rank * 3 * 3
f4_B = M * rank * 768
f4_ln = 768 * 2
t_A = M * 768 * rank
t_B = M * rank * 768
t_ln = 768 * 2
print(f"  - f3 adapters (A+dw+B): {f3_A + f3_dw + f3_B:,}")
print(f"  - f3 LN: {f3_ln:,}")
print(f"  - f4 adapters (A+dw+B): {f4_A + f4_dw + f4_B:,}")
print(f"  - f4 LN: {f4_ln:,}")
print(f"  - text adapters (A+B): {t_A + t_B:,}")
print(f"  - text LN: {t_ln:,}")

# 5. RegionBridge f3
rb_f3 = count_params(model.region_bridge_f3)
print(f"\n[RegionBridge f3] (R=12)")
print(f"  Total: {rb_f3:,}")

# 6. RegionBridge f4
rb_f4 = count_params(model.region_bridge_f4)
print(f"\n[RegionBridge f4] (R=6)")
print(f"  Total: {rb_f4:,}")

# RegionBridge breakdown
D = 192
text_dim = 768
num_heads = 4
num_refine = 2

for name, R, C in [("f3", 12, 384), ("f4", 6, 768)]:
    print(f"\n  [{name} detailed breakdown]")
    wq = text_dim * R * D + 0  # no bias in Linear by default? Let's check
    wf = C * D * 1 * 1 + D  # 1x1 conv with bias
    z_ln = D * 2
    
    # RefinementBlock x2
    self_ln = D * 2
    self_attn = D * D * 3 + D * D + D * 4  # QKV + out proj + biases
    cross_ln = D * 2
    kv_proj = text_dim * D + D
    cross_attn = D * D * 3 + D * D + D * 4
    ffn_ln = D * 2
    ffn = D * D * 4 + D * 4 + D * 4 * D + D  # linear1 + bias + linear2 + bias
    
    refine_total = num_refine * (self_ln + self_attn + cross_ln + kv_proj + cross_attn + ffn_ln + ffn)
    
    out_proj = D * C * 1 * 1 + C
    gamma = 1
    
    print(f"    wq: {wq:,} (768 → {R}*192)")
    print(f"    wf: {wf:,} ({C}→192 conv1x1)")
    print(f"    z_ln: {z_ln:,}")
    print(f"    refine_blocks x{num_refine}: {refine_total:,}")
    print(f"    out_proj: {out_proj:,}")
    print(f"    gamma: {gamma}")

# 7. Decoder
decoder = count_params(model.decoder)
print(f"\n[Decoder]")
print(f"  Total: {decoder:,}")

# Summary
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n" + "=" * 60)
print(f"TOTAL TRAINABLE: {total_trainable:,}")
print(f"Paper reports:   7,110,000")
print(f"Difference:      {total_trainable - 7110000:+,}")
print("=" * 60)

# List all modules by param count
print("\n[All modules sorted by param count]")
modules_params = []
for name, child in model.named_children():
    params = sum(p.numel() for p in child.parameters() if p.requires_grad)
    modules_params.append((name, params))

for name, params in sorted(modules_params, key=lambda x: -x[1]):
    print(f"  {name}: {params:,}")

# Check wq dimensions - this is likely the issue
print("\n[Checking wq in RegionBridge]")
print(f"  wq f3: input={model.region_bridge_f3.wq.in_features}, output={model.region_bridge_f3.wq.out_features}")
print(f"  wq f4: input={model.region_bridge_f4.wq.in_features}, output={model.region_bridge_f4.wq.out_features}")

# Paper says "768 → R×D" but we might have different interpretation
# If D=192, R=12 for f3: 768 → 12*192 = 768 → 2304
# If D=192, R=6 for f4: 768 → 6*192 = 768 → 1152
print(f"  Expected wq f3: 768 → {12*192} = {768 * 12 * 192:,} params")
print(f"  Expected wq f4: 768 → {6*192} = {768 * 6 * 192:,} params")
