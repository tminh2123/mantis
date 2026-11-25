# TMF-Seg Code vs. Slide Investigation Report

## Summary

The implementation broadly matches the slide descriptions, but there are **5 notable discrepancies** ranging from minor to major:

1. **Text encoder is trainable in code, but slides say frozen** (Slide 2)
2. **Image encoder has zero self-attention; slides claim attention at ≤64×64** (Slide 4) — MAJOR
3. **UNet encoder has 5 levels, not 4; last level lacks attention** (Slide 6)
4. **DCT channel attention uses 3 pooling operations (avg+max+min), slides show only 2** (Slide 5)
5. **Decoder outputs 3 channels; final 1-channel mask is derived by channel mean-pooling** (Slides 2, 6)

---

## Per-Slide Evidence

---

### Slide 2 — Overall Architecture

**Claim:** Image I → φ_θ (trainable AE encoder) → y_c B×4×32×32. Text T → BioBERT (frozen) → z_text B×10×768. Cross-attention: Q=Linear(4→512), K/V=Linear(768→512), 8 heads, attention_head_dim=64. → z_fused B×4×32×32. TGME → z_enhanced B×4×32×32. GT mask → E() (frozen) → y_0 → add noise → y_t. Concat(y_t, z_enh) → B×8×32×32 → UNet → ε̂ B×4×32×32 → D() (frozen) → Ŝ B×1×256×256.

#### ✅ Matches

| Claim | Evidence |
|---|---|
| φ_θ trainable | `cond_stage_trainable: true` in YAML; `ImageTextFusionEmbedder.image_encoder` is not frozen |
| y_c B×4×32×32 | `AEEncoderEmbedder`: Encoder outputs B×8×32×32 (double_z=True) → Conv2d(8→4,k=1) → B×4×32×32 |
| max_length=10 → z_text B×10×768 | `text_encoder_config.max_length: 10`; BERT last_hidden_state shape = B×10×768 |
| Cross-attn: 8 heads, dim_head=64 | `BasicTransformerBlock(dim=4, num_attention_heads=8, attention_head_dim=64, cross_attention_dim=768)` → inner_dim=512; to_q=Linear(4→512), to_k/to_v=Linear(768→512) |
| z_fused B×4×32×32 | Confirmed by reshape in `ImageTextFusionEmbedder.forward` |
| TGME present | `TextGuidedMultiFrequencyAttention` module enabled via `use_frequency_enhancement=True` |
| E() and D() frozen | `self.first_stage_model.train = disabled_train`; all params `requires_grad=False` |
| Concat(y_t, z_enh) → B×8×32×32 | UNet `in_channels: 8` in YAML |

#### ❌ Discrepancy 1 — Text encoder is NOT frozen

- **Slide:** "BioBERT (frozen)"
- **Config (`configs/SDSeg/Qatacovid19_text-ldm-kl-8.yaml`):**
  ```yaml
  text_encoder_config:
    model_name: emilyalsentzer/Bio_ClinicalBERT
    freeze: false
  ```
- **Code (`ldm/modules/encoders/modules.py`, `ImageTextFusionEmbedder.__init__`):**
  ```python
  if self.freeze_text:   # ← freeze_text = False → branch never taken
      self.text_encoder.eval()
      for param in self.text_encoder.parameters():
          param.requires_grad = False
  else:
      print(f"[ImageTextFusionEmbedder] Text encoder is trainable")
  ```
  ClinicalBERT is fully trainable. The slide's "frozen" label is wrong.

#### Minor naming note

The slide says "BioBERT"; the code loads `emilyalsentzer/Bio_ClinicalBERT`, which is a different (clinical domain) model.

#### ✅ Decoder output — corrected by post-processing

- **Slide:** Ŝ B×1×256×256.
- **Decoder config:** `out_ch: 3` → decoder outputs B×3×256×256.
- **Post-processing** (`SDSeg.py` ~line 1810):
  ```python
  x_samples_ddim = self.decode_first_stage(samples_pred[0])   # B×3×256×256
  x_samples_ddim = x_samples_ddim.mean(dim=1, keepdim=True)   # B×1×256×256
  ```
  The final mask IS B×1×256×256, but it is obtained by mean-averaging the 3 decoder output channels, not from a 1-channel decoder. The slide does not make this intermediate step clear.

---

### Slide 3 — TGME

**Claim:** z_fused → 3 DCT branches (high/mid/low freq), each with channel attention → weighted combination α_h·z_high + α_m·z_mid + α_l·z_low with softmax weights from text mean-pool → MLP.

#### ✅ Matches

| Claim | Evidence |
|---|---|
| 3 branches (high/mid/low) | `self.high_freq`, `self.mid_freq`, `self.low_freq` in `TextGuidedMultiFrequencyAttention` |
| Text mean-pool | `text_global = z_text.mean(dim=1)  # [B, 768]` |
| MLP → softmax weights (α_h, α_m, α_l) | `self.text_to_freq_weights = nn.Sequential(Linear(768,128), ReLU, Linear(128,3), Softmax)` |
| Weighted sum | `x_enhanced = freq[:, 0:1] * x_high + freq[:, 1:2] * x_mid + freq[:, 2:3] * x_low` |
| N=16/8/8 filters | `high_freq(frequency_branches=16)`, `mid_freq(8)`, `low_freq(8)` |

No discrepancies found for Slide 3 at the structural level.

---

### Slide 4 — Image Encoder

**Claim:** VAE-style conv encoder, 4 levels, channels 128→256→512→512, spatial 256→128→64→32→32, ResBlock×2 per level, **Self-Attn at resolutions ≤64×64**, middle block, final Conv1×1 → embed_dim=4.

#### ✅ Matches

| Claim | Evidence |
|---|---|
| ch=128, ch_mult=[1,2,4,4] → 128/256/512/512 | `image_encoder_config: ch: 128, ch_mult: [1, 2, 4, 4]` in YAML |
| num_res_blocks=2 | `num_res_blocks: 2` in YAML |
| Middle block (ResBlock→Attn→ResBlock) | `self.mid.block_1`, `self.mid.attn_1`, `self.mid.block_2` in `Encoder` class (`model.py`) |
| Final Conv1×1 → embed_dim=4 | `self.out = Conv2d(8, 4, 1)` in `AEEncoderEmbedder` |

#### ❌ Discrepancy 2 — No self-attention in image encoder (MAJOR)

- **Slide:** "Self-Attn at resolutions ≤64×64" (i.e., at 64×64 and 32×32).
- **Config:**
  ```yaml
  image_encoder_config:
    attn_resolutions: []   # ← empty list
  ```
- **Code (`model.py`, `Encoder.__init__`):**
  ```python
  if curr_res in attn_resolutions:  # attn_resolutions=[] → never True
      attn.append(make_attn(block_in, attn_type=attn_type))
  ```
  With `attn_resolutions: []`, **no self-attention is added at any resolution**. The slide is incorrect — the image encoder has no attention layers at all.

---

### Slide 5 — DCT Channel Attention

**Claim:** Each branch: x × dct_weight_k → AvgPool/MaxPool → sum/avg → B×4×1×1 → Conv1×1(4→2)→ReLU→Conv1×1(2→4) → sigmoid → channel attention.

#### ✅ Mostly matches

| Claim | Evidence |
|---|---|
| DCT filter multiplication | `x_pooled_spectral = x_pooled * params` per DCT weight buffer |
| AdaptiveAvgPool2d(1) + AdaptiveMaxPool2d(1) | `self.average_channel_pooling`, `self.max_channel_pooling` in `MultiFrequencyChannelAttention` |
| Conv1×1(4→2)→ReLU→Conv1×1(2→4) | With in_channels=4, reduction=2: `self.fc = Sequential(Conv2d(4,2,1), ReLU, Conv2d(2,4,1))` |
| Sigmoid | `F.sigmoid(...)` applied to combined map |

#### ❌ Discrepancy 3 — Code uses 3 pooling operations, not 2

- **Slide:** AvgPool/MaxPool only (2 operations combined).
- **Code (`modules.py`, `MultiFrequencyChannelAttention.forward`):**
  ```python
  multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
  multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
  multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)  # MinPool via negation
  # ...
  multi_spectral_attention_map = F.sigmoid(
      self.fc(avg_map) + self.fc(max_map) + self.fc(min_map)  # 3 separate fc outputs summed
  )
  ```
  The code adds a third pooling branch (min-pool via `-MaxPool(-x)`) and applies the `fc` MLP independently to each of the three pooled features, then sums all three before sigmoid. The slide describes only avg and max pooling.

---

### Slide 6 — Denoising U-Net

**Claim:** in=8, out=4, model_channels=192, channel_mult=[1,2,2,4,4], num_res_blocks=2, **attention at all resolutions**. Encoder: 192/384/384/768ch × 32/16/8/4×. Bottleneck: 768ch×2×2 ResBlock→Attn→ResBlock. Decoder mirrors encoder with skip connections. t_emb injected at every ResBlock.

#### ✅ Matches

| Claim | Evidence |
|---|---|
| in=8, out=4 | `in_channels: 8, out_channels: 4` in YAML |
| model_channels=192 | `model_channels: 192` in YAML |
| channel_mult=[1,2,2,4,4] | `channel_mult: [1, 2, 2, 4, 4]` in YAML |
| num_res_blocks=2 | `num_res_blocks: 2` in YAML |
| Bottleneck ResBlock→Attn→ResBlock at 2×2 | `self.middle_block = TimestepEmbedSequential(ResBlock, AttentionBlock, ResBlock)` |
| t_emb injected at every ResBlock | `ResBlock._forward` adds `emb_out` from `self.emb_layers(emb)` to activations |
| skip connections | `th.cat([h, hs.pop()], dim=1)` in decoder loop |

#### ❌ Discrepancy 4 — Slide shows 4 encoder levels; code has 5; last level lacks attention

- **Slide:** "Encoder: 192/384/384/768ch × 32/16/8/4×" — 4 levels. "Attention at all resolutions."
- **Code computation** with `channel_mult=[1,2,2,4,4]` and input 32×32:

| Level | mult | ch | spatial | ds | Attention? |
|---|---|---|---|---|---|
| 0 | 1 | 192 | 32×32 | 1 | ✅ (ds=1 ∈ [1,2,4,8]) |
| 1 | 2 | 384 | 16×16 | 2 | ✅ (ds=2) |
| 2 | 2 | 384 | 8×8 | 4 | ✅ (ds=4) |
| 3 | 4 | 768 | 4×4 | 8 | ✅ (ds=8) |
| **4** | **4** | **768** | **2×2** | **16** | **❌ (ds=16 ∉ [1,2,4,8])** |
| Middle | — | 768 | 2×2 | — | ✅ |

There are **5 encoder levels** (not 4). The 5th level (768ch × 2×2 at ds=16) has no attention. The slide collapses this 5th encoder level into the "Bottleneck" description and incorrectly claims attention at all encoder resolutions.

- **Code evidence** (`openaimodel.py`):
  ```python
  attention_resolutions: [1, 2, 4, 8]  # from YAML
  # Level 4: ds=16, not in attention_resolutions → AttentionBlock NOT added
  ```
  The final output layer also uses `model_channels` (192ch), not 768ch:
  ```python
  self.out = nn.Sequential(normalization(ch), nn.SiLU(),
                           zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)))
  ```

---

## Summary Table

| Slide | Issue | Severity | Fix recommendation |
|---|---|---|---|
| 2 | Text encoder labeled "frozen" but `freeze: false` in config | Medium | Update slide to say "trainable BioClinicalBERT" (or freeze it in code if desired) |
| 4 | "Self-Attn at ≤64×64" but `attn_resolutions: []` → no attention | **Major** | Either add `attn_resolutions: [32, 64]` to image encoder config, or remove the attention claim from slide |
| 5 | "AvgPool/MaxPool" but code uses AvgPool + MaxPool + MinPool | Minor | Update slide to show 3 pooling branches |
| 6 | Encoder shown as 4 levels; code has 5 (768ch × 2×2 without attention) | Medium | Update slide to show 5 levels, clarify that Level 5 has no attention |
| 2, 6 | Decoder output is 3ch; 1-channel mask obtained by channel mean | Minor | Clarify in slide that D() outputs 3ch → averaged to 1ch for binary mask |

---

## Conclusions

The core pipeline logic (VAE latent space, cross-attention fusion, TGME frequency weighting, diffusion noise schedule, DDIM sampling) is correctly described in the slides. The discrepancies are mostly in architectural details that affect the accuracy of the technical presentation:

1. **Most impactful**: The absence of self-attention in the image encoder (`attn_resolutions: []`) means the encoder slide (Slide 4) is wrong in a visually prominent way — the self-attention blocks shown should not be there.

2. **For the text encoder**: If the design intent is for ClinicalBERT to be frozen (as the slide states), set `freeze: true` in the config. The current trainable setting may actually be intentional for domain adaptation but contradicts the slide.

3. **UNet structure**: Adding the missing 5th encoder level (768ch × 2×2, no attention) to Slide 6 would make it accurate.
