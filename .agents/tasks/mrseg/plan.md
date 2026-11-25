# MRSeg Implementation Plan

## Findings from Exploration

### Iris pieces to copy/adapt
| Iris file | MRSeg target | Action |
|---|---|---|
| `iris/utils/__init__.py` | `MRSeg/utils.py` | Copy `set_seed`, `load_config`, `get_device`, `save_checkpoint`, `load_checkpoint` verbatim |
| `iris/utils/metrics.py` | `MRSeg/utils.py` | Copy `dice_score_per_sample`, `iou_score_per_sample`, `compute_metrics_per_sample` verbatim |
| `iris/models/losses.py` | `MRSeg/models/mrseg.py` or `utils.py` | Copy `DiceLoss`, `DiceBCELoss` verbatim; use `DiceBCELoss(bce_weight=1.0)` for equal BCE+Dice |
| `iris/run.py` | `MRSeg/run.py` | Reuse `AverageMeter`, `setup_logger`, `evaluate()`, `full_evaluate()`, `parse_args()`, `main()` structure; rewrite `build_model()`, `train()` param groups |
| `iris/datasets/dataset.py` | `MRSeg/datasets/qata.py` + `mosmed.py` | Adapt `IRISDataset`; switch to grayscale loading (Bx1xHxW); keep `_load_texts()` JSON/xlsx logic |

### Dataset formats confirmed

**QaTa-COV19 (c2seg format)** — from `note.txt` and `iris/configs/qata.yaml`:
```
c2seg/
├── train/
│   ├── images/     *.png CT slices
│   ├── masks/      mask_*.png
│   └── texts.json  {"Image0001.png": "multifocal GGO...", ...}
├── val/   (same)
└── test/  (same)
```
- Server path (from iris config, relative to `mantis/`): `../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/`
- Split counts: train 5,716 / val 1,429 / test 2,113
- `texts.json` key = image filename (e.g. `"Image0001.png"`); iris `IRISDataset._get_text()` already handles this

**MosMedData+ (LViT format)** — from `iris/configs/mosmed.yaml` and TMF-Seg loaders:
```
MosMedDataPlus/
├── Train_Folder/
│   ├── img/        *.png
│   ├── labelcol/   *.png
│   └── Train_text.xlsx   columns: 'filename', 'text_description'
├── Val_Folder/   (same, Val_text.xlsx)
└── Test_Folder/  (same, Test_text.xlsx)
```
- Server path: `../../dataset/MosMedDataPlus/Train_Folder` etc.
- iris `IRISDataset._find_dir(['images','img'])` already finds `img/`; `_find_dir(['masks','labelcol'])` finds `labelcol/`; `_load_texts()` already handles xlsx with column fallback

### HuggingFace ConvNeXt model internals
- Class: `transformers.ConvNextModel.from_pretrained("pretrained_models/convnext-tiny-224")`
- Call: `model(pixel_values, output_hidden_states=True)` → `outputs.hidden_states`
- `hidden_states` is a tuple with one tensor per stage, in channel-first format:
  - `hidden_states[0]`: after embeddings (patchify stem) → Bx96x56x56 = f1 (H/4)
  - `hidden_states[1]`: after stage 0 → Bx96x56x56 (same resolution, 3 blocks)
  - `hidden_states[2]`: after stage 1 → Bx192x28x28 = f2 (H/8)
  - `hidden_states[3]`: after stage 2 → Bx384x14x14 = f3 (H/16)
  - `hidden_states[4]`: after stage 3 → Bx768x7x7 = f4 (H/32)
- Use `hidden_states[1]` for f1 (96, H/4), `[2]` for f2 (192, H/8), `[3]` for f3 (384, H/16), `[4]` for f4 (768, H/32)
- **Verify** exact indices in `scripts/verify_model.py` by printing shapes; adjust if HF version differs
- Each `ConvNextLayer` (in `encoder.stages[i].layers[j]`) has:
  - `dwconv`: DW conv 7x7 (not LoRA target)
  - `layernorm`: LN, **make trainable**
  - `pwconv1`: `nn.Linear(C, 4C)`, **LoRA target** in stages[2,3]
  - `pwconv2`: `nn.Linear(4C, C)`, **LoRA target** in stages[2,3]
  - `act`: GELU
- Input to `pwconv1`/`pwconv2` is channel-last (B, H, W, C) — LoRA must handle this format

### HuggingFace BiomedBERT model internals
- Class: `transformers.AutoModel.from_pretrained("pretrained_models/pubmedbert")`
- Tokenizer: `transformers.AutoTokenizer.from_pretrained("pretrained_models/pubmedbert")`
- Call: `model(**tokens)` → `outputs.last_hidden_state` Bx24x768 = z_text
- Transformer layers: `model.encoder.layer[0..11]` (12 total)
- Last 4 layers: `model.encoder.layer[-4:]` = layers 8, 9, 10, 11
- Each attention layer: `layer.attention.self.query` and `layer.attention.self.value` are `nn.Linear(768, 768)` — **LoRA target**
- LayerNorm params: `layer.attention.output.LayerNorm`, `layer.output.LayerNorm` — **make trainable**

---

## Tensor Shape Walkthrough

| Step | Module | Input shape | Output shape |
|---|---|---|---|
| 0 | Dataset | — | image: Bx1x224x224, mask: Bx1x224x224, text: list[str] |
| 1a | MedImageAdapter | Bx1x224x224 | I3: Bx3x224x224 |
| 1b | I3 stem layer 1 | I3: Bx3x224x224 | skip1: Bx32x112x112 (H/2) |
| 1c | I3 stem layer 2 | I3: Bx3x224x224 | skip2: Bx16x224x224 (H) |
| 2 | ConvNeXtWithLoRA | I3: Bx3x224x224 | f1:Bx96x56x56, f2:Bx192x28x28, f3:Bx384x14x14, f4:Bx768x7x7 |
| 3 | PubMedBERTWithLoRA | texts list len B | z_text: Bx24x768 |
| 4 | Router | f4+z_text | alpha: Bx4 sparse |
| 5 | PairAdapter | f3,f4,z_text,alpha | f3':Bx384x14x14, f4':Bx768x7x7, t':Bx24x768 |
| 6a | RegionBridge(f3') | f3':Bx384x14x14, t':Bx24x768 | f3'':Bx384x14x14 |
| 6b | RegionBridge(f4') | f4':Bx768x7x7, t':Bx24x768 | f4'':Bx768x7x7 |
| 7 | MRSegDecoder | f1..f4'', skip1, skip2 | pred: Bx1x224x224 |

**RegionBridge(f3') internals (R3=12, D=192, N=14×14=196):**
1. Q = reshape(Linear(768→12×192)(Mean(t'))): Bx768 → Bx2304 → Bx12x192
2. Z = reshape(Conv1x1(384→192)(f3')): Bx192x14x14 → Bx196x192
3. LN(Z): Bx196x192
4. A = softmax_N(Q × LN(Z)ᵀ / √192): Bx12x196
5. G = A × Z: Bx12x192
6. 2× [SelfAttn(G) + CrossAttn(G queries, t' keys+vals projected to D) + FFN]: Bx12x192 → G_hat
7. F' = F + γ × Conv1x1(192→384, zero-init)(reshape(Aᵀ × G_hat)): Bx384x14x14

**RegionBridge(f4') internals (R4=6, D=192, N=7×7=49):**
- Same as above but R=6, N=49, C=768

**MRSegDecoder internals:**
```
f4'' (768,7)  →proj4(768→192)→  (192,7)  →Up2x→  (192,14)
                                                    + proj3(384→192)(f3'') = x1 (192,14)
x1 (192,14)  →Up2x→  (192,28)  →gated_fuse(f2 192,28)=  x2 (192,28)
x2 (192,28)  →Up2x→  (192,56)  →gated_fuse(proj1(96→192)(f1))=  x3 (192,56)
x3 (192,56)  →Up2x→  (192,112) + proj_skip1(32→192)(skip1)=  x4 (192,112)
x4 (192,112) →Up2x→  (192,224) + proj_skip2(16→192)(skip2)=  x5 (192,224)
x5           →coarse: Conv1x1(192→1) + detail: DWConv3x3→Conv1x1(192→1)= Y (1,224)
```

**Gated fuse formula** (applied at f2 and proj1(f1)):
```
gate = tanh(G(cat[Up(x), f]))   # G = Conv2d(384→192, 1×1)
f_tilde = f + f * gate           # gated residual on skip feature
x_out = Up(x) + f_tilde          # merge upsampled and gated skip
```

---

## Implementation Plan

- [ ] 1. Create directory structure and empty `__init__.py` files
      Make all directories and stubs so `import` resolves everywhere.
      Files to create:
      - `MRSeg/models/__init__.py`
      - `MRSeg/datasets/__init__.py`
      - `MRSeg/configs/` (directory)
      - `MRSeg/scripts/` (directory)
      - `MRSeg/output/checkpoints/`, `output/logs/`, `output/analysis/` (directories)
      Verify: `python -c "import sys; sys.path.insert(0,'MRSeg'); from models import __init__"` exits 0 from `computer-vision/`.

- [ ] 2. Create `MRSeg/utils.py`
      Copy `set_seed`, `load_config`, `get_device`, `save_checkpoint`, `load_checkpoint` from `iris/utils/__init__.py`; copy `dice_score_per_sample`, `iou_score_per_sample`, `compute_metrics_per_sample` from `iris/utils/metrics.py`; copy `DiceLoss`, `DiceBCELoss` from `iris/models/losses.py`. Add `AverageMeter` class (copy from iris/run.py). No modifications needed.
      Files: `MRSeg/utils.py`
      Verify: `python -m py_compile MRSeg/utils.py` from `computer-vision/`.

- [ ] 3. Create `MRSeg/models/lora.py`
      **Decision:** Two LoRA classes share the same init signature (`in_features, out_features, rank=8, scale=16, dropout=0.05`) but differ in the low-rank path.

      **`LoRALinear`** (for BERT Q, V):
      - `A = nn.Linear(in_features, rank, bias=False)` — kaiming init
      - `B = nn.Linear(rank, out_features, bias=False)` — zero-init
      - `dropout = nn.Dropout(dropout)`
      - `forward(x)`: `self.original(x) + (scale/rank) * B(dropout(A(x)))`
      - `self.original` is set externally after wrapping

      **`LoRALinearWithDW`** (for ConvNeXt pwconv1/pwconv2, input is B×H×W×C channel-last):
      - Same A, B, dropout
      - `dw = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)` — kaiming init
      - `forward(x)`: x is (B,H,W,C); low-rank path: A(x)→(B,H,W,r) permute to (B,r,H,W) → dw → permute back (B,H,W,r) → B → (B,H,W,C); add to original(x)
      - `self.original` is set externally

      Helper `apply_lora_to_linear(module, attr_name, lora_cls, **lora_kwargs)`:
      - Gets the original `nn.Linear` from `module.attr_name`
      - Creates the LoRA wrapper, sets `wrapper.original = original_linear`
      - Replaces `module.attr_name` with wrapper
      - Keeps original frozen: `for p in original_linear.parameters(): p.requires_grad_(False)`

      Files: `MRSeg/models/lora.py`
      Verify: `python -m py_compile MRSeg/models/lora.py`

- [ ] 4. Create `MRSeg/models/encoders.py`
      **`ConvNeXtWithLoRA(nn.Module)`:**
      - `__init__(pretrained_path, lora_rank=8, lora_scale=16, lora_dropout=0.05)`:
        1. `self.model = ConvNextModel.from_pretrained(pretrained_path)`
        2. Freeze all: `for p in self.model.parameters(): p.requires_grad_(False)`
        3. Apply LoRA to stages[2] and stages[3]: iterate `self.model.encoder.stages[2].layers` and `stages[3].layers`; for each layer call `apply_lora_to_linear(layer, 'pwconv1', LoRALinearWithDW, ...)` and `apply_lora_to_linear(layer, 'pwconv2', LoRALinearWithDW, ...)`
        4. Make LayerNorms trainable: iterate stages[2,3] layers, set `layer.layernorm.weight.requires_grad_(True)` and `layer.layernorm.bias.requires_grad_(True)` (also the downsampling LayerNorm if present)
      - `forward(x)` where x is Bx3xHxW:
        1. `outputs = self.model(x, output_hidden_states=True)`
        2. `hs = outputs.hidden_states`  # tuple of 5 for ConvNeXt-Tiny
        3. `f1=hs[1], f2=hs[2], f3=hs[3], f4=hs[4]`  # see notes — verify shapes on server
        4. Return `f1, f2, f3, f4`

      **`PubMedBERTWithLoRA(nn.Module)`:**
      - `__init__(pretrained_path, max_length=24, lora_rank=8, lora_scale=16, lora_dropout=0.05)`:
        1. `self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)`
        2. `self.model = AutoModel.from_pretrained(pretrained_path)`
        3. Freeze all params
        4. Apply LoRA to Q and V of last 4 layers: `for layer in self.model.encoder.layer[-4:]`: call `apply_lora_to_linear(layer.attention.self, 'query', LoRALinear, ...)` and `apply_lora_to_linear(layer.attention.self, 'value', LoRALinear, ...)`
        5. Make LayerNorm params trainable in those same 4 layers: `layer.attention.output.LayerNorm`, `layer.output.LayerNorm`
      - `forward(texts: List[str], device) -> Tensor Bx24x768`:
        1. Tokenize with `padding="max_length", truncation=True, max_length=self.max_length`; move to device
        2. `out = self.model(**tokens)`
        3. Return `out.last_hidden_state`  # Bx24x768

      Files: `MRSeg/models/encoders.py`
      Verify: `python -m py_compile MRSeg/models/encoders.py`

- [ ] 5. Create `MRSeg/models/adapters.py`
      Three components: `MedImageAdapter`, `Router`, `PairAdapter`.

      **`MedImageAdapter(nn.Module)`:**
      - `__init__(adapter_channels=64)`:
        - `a_conv1 = nn.Conv2d(1, 64, 1)` — Conv1x1(1→C)
        - `a_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64)` — DWConv3x3
        - `a_norm = nn.GroupNorm(1, 64)` — or LayerNorm; use GroupNorm(num_groups=1) = InstanceNorm
        - `a_act = nn.GELU()`
        - `a_conv2 = nn.Conv2d(64, 3, 1)` — Conv1x1(C→3), **zero-init weight and bias**
        - `stem_half = nn.Sequential(Conv2d(3,32,3,stride=2,pad=1), GroupNorm(4,32), GELU())` — H→H/2, produces skip1 (32ch)
        - `stem_full = nn.Sequential(Conv2d(3,16,3,stride=1,pad=1), GroupNorm(4,16), GELU())` — H→H, produces skip2 (16ch)
      - `forward(x_gray)` where x_gray is Bx1xHxW:
        - `i_base = x_gray.repeat(1, 3, 1, 1)`
        - `a_out = a_conv2(a_act(a_norm(a_dw(a_conv1(x_gray)))))`  # Bx3xHxW
        - `i3 = i_base + a_out`
        - `skip1 = stem_half(i3)`  # Bx32xH/2xW/2
        - `skip2 = stem_full(i3)`  # Bx16xHxW
        - Return `i3, skip1, skip2`

      **`Router(nn.Module)`:**
      - `__init__(visual_dim=768, text_dim=768, num_adapters=4, topk=2)`:
        - `mlp = nn.Sequential(Linear(1536, 768), GELU(), Linear(768, 4))`
        - `topk=2`, `num_adapters=4`
      - `forward(f4, z_text)` → alpha Bx4:
        - `v_pool = F.adaptive_avg_pool2d(f4, 1).flatten(1)`  # Bx768
        - `t_mean = z_text.mean(1)`  # Bx768
        - `r = mlp(cat([v_pool, t_mean], dim=1))`  # Bx4
        - `s = F.softmax(r, dim=1)`  # Bx4
        - **TopKNorm**: for each sample, zero out the bottom (4-topk)=2 values, then renormalize the top-2: `topk_vals, topk_idx = s.topk(2, dim=1)`; create sparse alpha by scattering; renormalize (divide by sum of kept values)
        - Return alpha Bx4

      **`PairAdapter(nn.Module)`:**
      - `__init__(f3_channels=384, f4_channels=768, text_dim=768, num_adapters=4, rank=8, eta=16)`:
        - For visual banks (f3 and f4): each bank has M=4 sets of (A_m, DWConv, B_m); A_m=Linear(C→r), DW=Conv2d(r,r,3,pad=1,groups=r), B_m=Linear(r→C); zero-init all B_m; LN for visual features
        - For text bank: M=4 sets of (A_m, B_m); A_m=Linear(768→r), B_m=Linear(r→768); zero-init all B_m; LN for text
        - `self.f3_lns = nn.ModuleList([nn.LayerNorm(f3_channels)])`  — single LN applied before all adapters
        - `self.f4_lns = nn.ModuleList([nn.LayerNorm(f4_channels)])`
        - `self.t_ln = nn.LayerNorm(text_dim)`
        - `self.eta = eta`, `self.rank = rank`
      - `_apply_visual_bank(h, bank_a, bank_dw, bank_b, alpha, ln)` where h is BxCxHxW:
        - Flatten: BxCxN → transpose → BxNxC
        - LN: apply ln
        - For each m in 0..3: `out_m = B_m(dw_m(A_m(h_ln)).view...)` with spatial reshape for DW
        - `h' = h + (eta/rank) * sum_m alpha[:,m:m+1,None]*out_m`
        - Actually more precisely: flatten h to BxNxC; `h_ln = ln(h_flat)` BxNxC; for each m: `a = A_m(h_ln)` BxNxr; reshape to Bxrx√NxN for DW; apply DW; flatten back; `b = B_m(result)` BxNxC; weighted sum with alpha
      - `_apply_text_bank(t, bank_a, bank_b, alpha, ln)` where t is Bx24x768:
        - `t_ln = ln(t)` Bx24x768
        - For each m: `a = A_m(t_ln)` Bx24xr; `b = B_m(a)` Bx24x768
        - `t' = t + (eta/rank) * sum_m alpha[:,m:m+1,None]*b`  # alpha broadcast over L=24
      - `forward(f3, f4, z_text, alpha)` → f3', f4', t'

      Files: `MRSeg/models/adapters.py`
      Verify: `python -m py_compile MRSeg/models/adapters.py`

- [ ] 6. Create `MRSeg/models/region_bridge.py`
      **`RefinementBlock(nn.Module)`** — single self+cross+FFN block:
      - `__init__(dim=192, text_dim=768, num_heads=4)`:
        - `self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)`
        - `self_ln = nn.LayerNorm(dim)`
        - `kv_proj = nn.Linear(text_dim, dim)`  — project t' to D for cross-attn K,V
        - `cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)`
        - `cross_ln = nn.LayerNorm(dim)`
        - `ffn = nn.Sequential(Linear(dim, dim*4), GELU(), Linear(dim*4, dim))`
        - `ffn_ln = nn.LayerNorm(dim)`
      - `forward(g, t_prime)`:
        - g: BxRxD, t_prime: Bx24x768
        - Self-attn: `g = g + self_attn(self_ln(g), self_ln(g), self_ln(g))[0]`
        - Cross-attn: kv = kv_proj(t_prime) BxLxD; `g = g + cross_attn(cross_ln(g), kv, kv)[0]`
        - FFN: `g = g + ffn(ffn_ln(g))`
        - Return g

      **`RegionBridge(nn.Module)`**:
      - `__init__(visual_channels, num_regions, D=192, text_dim=768, num_heads=4, num_refine=2)`:
        - `wq = nn.Linear(text_dim, num_regions * D)`
        - `wf = nn.Conv2d(visual_channels, D, 1)`
        - `z_ln = nn.LayerNorm(D)`
        - `refine_blocks = nn.ModuleList([RefinementBlock(D, text_dim, num_heads) for _ in range(num_refine)])`
        - `out_proj = nn.Conv2d(D, visual_channels, 1)`  — **zero-init weight and bias**
        - `gamma = nn.Parameter(torch.ones(1) * 0.1)`
        - `D = D`, `R = num_regions`
      - `forward(F_in, t_prime)`:
        1. B, C, H, W = F_in.shape; N = H*W
        2. Q: `mean_t = t_prime.mean(1)` Bx768; `Q = wq(mean_t).view(B, R, D)` BxRxD
        3. Z: `Z = wf(F_in).flatten(2).permute(0,2,1)` BxNxD
        4. Z_ln = z_ln(Z) BxNxD
        5. A = softmax(Q @ Z_ln.transpose(1,2) / sqrt(D), dim=-1) BxRxN  — softmax over N dimension
        6. G = A @ Z BxRxD
        7. For each refine_block: G = block(G, t_prime)  → G_hat BxRxD
        8. recon = (A.transpose(1,2) @ G_hat)  BxNxD → permute(0,2,1).view(B,D,H,W)
        9. delta = out_proj(recon) BxCxHxW
        10. F_out = F_in + gamma * delta
        11. Return F_out

      Files: `MRSeg/models/region_bridge.py`
      Verify: `python -m py_compile MRSeg/models/region_bridge.py`

- [ ] 7. Create `MRSeg/models/decoder.py`
      **`GatedFuse(nn.Module)`** — the gated skip connection block:
      - `__init__(up_channels=192, skip_channels=192)`:
        - `skip_proj = nn.Conv2d(skip_channels, up_channels, 1)` if `skip_channels != up_channels` else `nn.Identity()`
        - `gate_conv = nn.Conv2d(up_channels * 2, up_channels, 1)`
      - `forward(x_up, f_skip)`:
        - `f = skip_proj(f_skip)` → BxCxHxW
        - `gate = torch.tanh(gate_conv(cat([x_up, f], dim=1)))`
        - `f_tilde = f + f * gate`
        - Return `x_up + f_tilde`

      **`MRSegDecoder(nn.Module)`**:
      - `__init__(dim=192)`:
        - `proj4 = nn.Conv2d(768, dim, 1)`
        - `proj3 = nn.Conv2d(384, dim, 1)`
        - `proj1 = nn.Conv2d(96, dim, 1)`   — project f1 to decoder dim
        - `proj_skip1 = nn.Conv2d(32, dim, 1)`  — project skip1 (H/2)
        - `proj_skip2 = nn.Conv2d(16, dim, 1)`  — project skip2 (H)
        - `gated2 = GatedFuse(dim, dim)`   — fuses f2 (192ch matches decoder dim directly, no proj needed)
        - `gated1 = GatedFuse(dim, dim)`   — fuses proj1(f1)
        - `up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)` (shared)
        - `coarse_head = nn.Conv2d(dim, 1, 1)`
        - `detail_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)`
        - `detail_head = nn.Conv2d(dim, 1, 1)`
      - `forward(f1, f2, f3pp, f4pp, skip1, skip2)`:
        - `x1 = up(proj4(f4pp)) + proj3(f3pp)`  # (192, H/16)
        - `x2 = gated2(up(x1), f2)`  # (192, H/8); f2 is 192ch so GatedFuse uses no proj
        - `x3 = gated1(up(x2), proj1(f1))`  # (192, H/4)
        - `x4 = up(x3) + proj_skip1(skip1)`  # (192, H/2)
        - `x5 = up(x4) + proj_skip2(skip2)`  # (192, H)
        - `coarse = coarse_head(x5)` Bx1xHxW
        - `detail = detail_head(detail_dw(x5))` Bx1xHxW
        - Return `coarse + detail`

      Files: `MRSeg/models/decoder.py`
      Verify: `python -m py_compile MRSeg/models/decoder.py`

- [ ] 8. Create `MRSeg/models/mrseg.py` — the top-level model
      **`MRSegModel(nn.Module)`**:
      - `__init__(convnext_path="pretrained_models/convnext-tiny-224", bert_path="pretrained_models/pubmedbert", max_token_len=24, lora_rank=8, lora_scale=16, lora_dropout=0.05, decoder_dim=192, num_regions_f3=12, num_regions_f4=6, router_topk=2, router_num_adapters=4, pair_adapter_rank=8, pair_adapter_eta=16, img_adapter_channels=64)`:
        1. `self.med_adapter = MedImageAdapter(img_adapter_channels)`
        2. `self.visual_enc = ConvNeXtWithLoRA(convnext_path, lora_rank, lora_scale, lora_dropout)`
        3. `self.text_enc = PubMedBERTWithLoRA(bert_path, max_token_len, lora_rank, lora_scale, lora_dropout)`
        4. `self.router = Router(768, 768, router_num_adapters, router_topk)`
        5. `self.pair_adapter = PairAdapter(384, 768, 768, router_num_adapters, pair_adapter_rank, pair_adapter_eta)`
        6. `self.region_bridge_f3 = RegionBridge(384, num_regions_f3, decoder_dim)`
        7. `self.region_bridge_f4 = RegionBridge(768, num_regions_f4, decoder_dim)`
        8. `self.decoder = MRSegDecoder(decoder_dim)`

      - `forward(x, texts)` where x is Bx1xHxW (or Bx3xHxW — convert to grayscale if 3-ch):
        - If x.shape[1] == 3: `x = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]`  # RGB→gray
        - `i3, skip1, skip2 = self.med_adapter(x)`
        - `f1, f2, f3, f4 = self.visual_enc(i3)`
        - `z_text = self.text_enc(texts, x.device)`
        - `alpha = self.router(f4, z_text)`
        - `f3p, f4p, t_prime = self.pair_adapter(f3, f4, z_text, alpha)`
        - `f3pp = self.region_bridge_f3(f3p, t_prime)`
        - `f4pp = self.region_bridge_f4(f4p, t_prime)`
        - `pred = self.decoder(f1, f2, f3pp, f4pp, skip1, skip2)`  # Bx1xHxW
        - Return `{'pred': pred}`

      **`build_mrseg(cfg) -> MRSegModel`** factory:
      - Reads `cfg['model']` dict and passes all keys to `MRSegModel.__init__`

      **`__init__.py` for models**:
      - Export `MRSegModel`, `build_mrseg`, `DiceBCELoss`

      Files: `MRSeg/models/mrseg.py`, `MRSeg/models/__init__.py`
      Verify: `python -m py_compile MRSeg/models/mrseg.py`

- [ ] 9. Create `MRSeg/datasets/qata.py` and `MRSeg/datasets/mosmed.py`
      **Decision:** Adapt `IRISDataset` from iris. Only change is `_load_image()` returns grayscale (Bx1xHxW) instead of RGB. The `_load_texts()`, `_find_dir()`, `_find_mask()`, `_get_text()`, augmentation, and DataLoader wrapper are **copied verbatim**.

      **`MRSegDataset(Dataset)`** — shared base in a common module:
      - `__init__(root_dir, image_size=224, augment=False)`: same as `IRISDataset`
      - `__getitem__`: replace `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` with `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)` and `torch.from_numpy(image).unsqueeze(0).float() / 255.0`  → Bx1xHxW
      - Everything else identical to `IRISDataset`

      **`qata.py`**: import and re-export `MRSegDataset` with a `QaTaDataset = MRSegDataset` alias.

      **`mosmed.py`**: import and re-export `MRSegDataset` with a `MosMedDataset = MRSegDataset` alias.
      (Both datasets have the same on-disk format handled by the shared base.)

      **`datasets/__init__.py`**:
      ```python
      from .qata import QaTaDataset
      from .mosmed import MosMedDataset
      from .dataset import MRSegDataset, get_dataloader
      ```

      `get_dataloader(root_dir, batch_size, image_size=224, augment=False, shuffle=True, num_workers=4)` → same pattern as iris.

      Files: `MRSeg/datasets/dataset.py`, `MRSeg/datasets/qata.py`, `MRSeg/datasets/mosmed.py`, `MRSeg/datasets/__init__.py`
      Verify: `python -m py_compile MRSeg/datasets/dataset.py`

- [ ] 10. Create `MRSeg/configs/qata.yaml` and `MRSeg/configs/mosmed.yaml`
       **Config key design** — mirrors iris pattern so `run.py` can use the same `load_config` → dict access pattern.

       **`configs/qata.yaml`**:
       ```yaml
       project:
         name: mrseg
         dataset: qata
         seed: 42

       data:
         # Paths relative to run directory (mantis/)
         train_dir: "../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/train"
         val_dir:   "../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/val"
         test_dir:  "../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/test"
         image_size: 224
         num_workers: 4

       model:
         convnext_path: "MRSeg/pretrained_models/convnext-tiny-224"
         bert_path:     "MRSeg/pretrained_models/pubmedbert"
         max_token_len: 24
         lora_rank:     8
         lora_scale:    16
         lora_dropout:  0.05
         decoder_dim:   192
         num_regions_f3: 12
         num_regions_f4: 6
         router_topk:    2
         router_num_adapters: 4
         pair_adapter_rank: 8
         pair_adapter_eta:  16
         img_adapter_channels: 64

       train:
         epochs:        200
         batch_size:    16
         lr:            1.0e-3
         lora_lr:       1.0e-4
         weight_decay:  1.0e-4
         bce_weight:    1.0
         grad_clip:     1.0
         patience:      30
         cosine_lr:     true
       ```

       **`configs/mosmed.yaml`**: same structure; change `data.*_dir` to MosMedDataPlus paths:
       ```yaml
       data:
         train_dir: "../../dataset/MosMedDataPlus/Train_Folder"
         val_dir:   "../../dataset/MosMedDataPlus/Val_Folder"
         test_dir:  "../../dataset/MosMedDataPlus/Test_Folder"
       ```
       Also set `project.dataset: mosmed` and `train.lr: 3.0e-4` (MosMed is smaller, lower LR).

       Files: `MRSeg/configs/qata.yaml`, `MRSeg/configs/mosmed.yaml`
       Verify: `python -c "import yaml; yaml.safe_load(open('MRSeg/configs/qata.yaml'))"` from `computer-vision/`.

- [ ] 11. Create `MRSeg/run.py`
       Adapt `iris/run.py`. Changes from iris:
       1. Import `MRSegModel, build_mrseg, DiceBCELoss` instead of iris classes
       2. `build_model(cfg, device)`: call `build_mrseg(cfg)`
       3. `train()` optimizer param groups — split by name prefix:
          - `lora` params (LoRA A/B/dw in encoders): `cfg['train']['lora_lr']`
          - `adapter` params (med_adapter, pair_adapter, router): full `cfg['train']['lr']`
          - `decoder` params (region_bridge, decoder): full `cfg['train']['lr']`
          - **Empty groups are dropped** (same guard as iris)
       4. Loss: `criterion = DiceBCELoss(bce_weight=cfg['train']['bce_weight'])`  — no deep supervision
       5. `outputs = model(images, texts)` → `loss = criterion(outputs['pred'], masks)`
       6. `parse_args()`: drop `--visual`, `--text`, `--fusion`, `--pretrained`, `--freeze`, `--num_heads`, `--expansion`, `--memory_size`; keep `--config`, `--tag`, `--mode`, `--lr`, `--epochs`, `--patience`, `--batch_size`, `--checkpoint`, `--output_dir`
       7. Log line: `"Model: MRSeg, regions=({R3},{R4}), lora_rank={rank}"`

       Everything else (AverageMeter, setup_logger, evaluate, full_evaluate, scheduler, checkpointing) **copied verbatim** from iris/run.py.

       Files: `MRSeg/run.py`
       Verify: `python -m py_compile MRSeg/run.py`

- [ ] 12. Create `MRSeg/scripts/download_pretrained.sh`
       ```bash
       #!/usr/bin/env bash
       # Run from MRSeg/ directory
       mkdir -p pretrained_models

       echo "Downloading ConvNeXt-Tiny..."
       huggingface-cli download facebook/convnext-tiny-224 \
           --local-dir pretrained_models/convnext-tiny-224

       echo "Downloading BiomedBERT..."
       huggingface-cli download microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
           --local-dir pretrained_models/pubmedbert

       echo "Done. Pretrained models in MRSeg/pretrained_models/"
       ```
       Files: `MRSeg/scripts/download_pretrained.sh`
       Verify: `bash -n MRSeg/scripts/download_pretrained.sh` (syntax check only).

- [ ] 13. Create `MRSeg/scripts/verify_model.py`
       Run on server to validate model before training. Covers the cases where torch/transformers are not available locally.
       ```python
       """
       Run from mantis/ on server:
           python MRSeg/scripts/verify_model.py
       """
       import sys
       sys.path.insert(0, 'MRSeg')

       import torch
       from models.mrseg import MRSegModel

       print("Building model...")
       m = MRSegModel(
           convnext_path="MRSeg/pretrained_models/convnext-tiny-224",
           bert_path="MRSeg/pretrained_models/pubmedbert",
       )
       m.eval()
       print("Model built OK")

       # Print trainable parameter count
       n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
       n_total = sum(p.numel() for p in m.parameters())
       print(f"Params: {n_trainable:,} trainable / {n_total:,} total")

       # Forward pass
       B = 2
       x = torch.randn(B, 1, 224, 224)
       texts = ['infection in lower lobe', 'ground-glass opacity bilateral']
       print("Running forward pass...")
       with torch.no_grad():
           out = m(x, texts)
       pred = out['pred']
       print(f"Output shape: {pred.shape}")  # expect (2, 1, 224, 224)
       assert pred.shape == (B, 1, 224, 224), f"Wrong shape: {pred.shape}"

       # Loss + backward
       from utils import DiceBCELoss
       criterion = DiceBCELoss(bce_weight=1.0)
       mask = torch.randint(0, 2, (B, 1, 224, 224)).float()
       m.train()
       out2 = m(x, texts)
       loss = criterion(out2['pred'], mask)
       print(f"Loss: {loss.item():.4f}")
       loss.backward()
       print("Backward OK")

       # Print ConvNeXt hidden state shapes to verify indices
       print("\nConvNeXt hidden state debug:")
       from models.encoders import ConvNeXtWithLoRA
       enc = m.visual_enc
       i3 = torch.randn(1, 3, 224, 224)
       with torch.no_grad():
           hs = enc.model(i3, output_hidden_states=True).hidden_states
       for i, h in enumerate(hs):
           print(f"  hidden_states[{i}]: {h.shape}")

       print("\n=== ALL CHECKS PASSED ===")
       ```
       Files: `MRSeg/scripts/verify_model.py`
       Verify: `python -m py_compile MRSeg/scripts/verify_model.py`

- [ ] 14. Local syntax verification (no torch required)
       Compile every Python file with `py_compile` to catch syntax errors before pushing.
       ```bash
       python -m py_compile \
           MRSeg/utils.py \
           MRSeg/models/lora.py \
           MRSeg/models/encoders.py \
           MRSeg/models/adapters.py \
           MRSeg/models/region_bridge.py \
           MRSeg/models/decoder.py \
           MRSeg/models/mrseg.py \
           MRSeg/datasets/dataset.py \
           MRSeg/run.py \
           MRSeg/scripts/verify_model.py
       ```
       All must exit 0. If torch is available locally (`.venv`), additionally run:
       ```bash
       cd /Users/ntminh3/workspace/computer-vision
       .venv/bin/python -c "
       import sys; sys.path.insert(0,'MRSeg')
       from models.mrseg import MRSegModel
       print('Import OK')
       "
       ```
       Files: none (verification step only)
       Verify: all `py_compile` commands exit 0.

---

## Server Run Commands

**Setup** (run once from server):
```bash
cd /dscilab_hvch261_tuanminh/workspace/code/mantis/MRSeg
bash scripts/download_pretrained.sh
python scripts/verify_model.py          # run from mantis/
```

**Training — QaTa-COV19:**
```bash
cd /dscilab_hvch261_tuanminh/workspace/code/mantis
python MRSeg/run.py \
    --config MRSeg/configs/qata.yaml \
    --tag mrseg_v1 \
    --output_dir MRSeg/output
```

**Training — MosMedData+:**
```bash
cd /dscilab_hvch261_tuanminh/workspace/code/mantis
python MRSeg/run.py \
    --config MRSeg/configs/mosmed.yaml \
    --tag mrseg_v1 \
    --output_dir MRSeg/output
```

**Evaluation only:**
```bash
python MRSeg/run.py \
    --config MRSeg/configs/qata.yaml \
    --tag mrseg_v1 \
    --mode eval \
    --checkpoint MRSeg/output/checkpoints/qata_mrseg_v1/best.pth \
    --output_dir MRSeg/output
```

Checkpoints land at `MRSeg/output/checkpoints/{dataset}_{tag}/best.pth`.
Logs land at `MRSeg/output/logs/{dataset}_{tag}.log`.

---

## Architecture Decisions & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Dataset returns | Bx1xHxW grayscale | Architecture spec shows 1-ch `I_gray` input; simpler than in-model conversion |
| ConvNeXt output | `output_hidden_states=True`, indices [1],[2],[3],[4] | HF ConvNextModel documents this interface; verify_model.py prints actual shapes |
| LoRA ConvNeXt format | `LoRALinearWithDW` handles (B,H,W,C) channel-last | HF ConvNeXt uses NHWC before pwconv; reshape to NCHW only for DW3x3, then back |
| LoRA BERT format | `LoRALinear` standard (B,L,C) | BERT Q/V projections operate on sequence tokens, no spatial mixing needed |
| Norm trainability | Only LN in LoRA'd layers (stages[2,3] for ConvNeXt, last 4 layers for BERT) | Spec says "Norm params trainable"; restricting scope prevents over-fitting frozen layers |
| RegionBridge cross-attn | G as query, t' projected to D as K and V | Spec note says "use standard MHA with G as query and t' as key/value"; D=192 reduces BERT dim |
| I3 stem channels | skip1=32ch, skip2=16ch, both projected to 192 in decoder via 1x1 | Keeps stem lightweight (<1% params); decoder projection is negligible |
| Loss | `DiceBCELoss(bce_weight=1.0)` | Spec: "BCE + SoftDice, equal weight"; `bce_weight=1.0` in iris code means equal 1.0×Dice + 1.0×BCE |
| PairAdapter LN scope | Single shared LN per modality bank (not per adapter) | Reduces parameters; all M=4 adapters operate on same normalized representation |
| Optimizer param groups | 3 groups: lora_lr, adapter_lr, decoder_lr | LoRA params need lower LR to stay close to pretrained; adapters and decoder are trained from scratch |
