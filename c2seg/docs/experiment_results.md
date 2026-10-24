# C2Seg Experiment Results & Tag Mapping

## Tag Naming Convention

Format: `{dataset}_{source}_{variant}`

- **dataset**: `qata` | `mosmed`
- **source**: `custom` (tự impl) | `ukan` (U-KAN block) | `c2seg` (code gốc paper)
- **variant**: `kan` | `mlp` | `kan_lite` | `deepsup` | `full` | `0.5` (scale)

---

## Tag Mapping (Old → New)

### QaTa-COV19 Dataset

| Old Tag | New Tag | Description |
|---------|---------|-------------|
| `efficient_kan_v1` | `qata_custom_kan` | Custom impl theo paper, B-spline KAN |
| `simple_kan_v1` | `qata_custom_kan_lite` | Bỏ base spline để giảm OOM |
| `ukan_0.5` | `qata_ukan_0.5` | U-KAN block, scale 0.5 |
| `mlp_1.0` | `qata_custom_mlp` | Custom impl, full MLP thay KAN |
| `c2seg_kan` | `qata_c2seg_kan` | C2Seg origin + KAN K-Gate |
| `c2seg_kan_0.5` | `qata_c2seg_kan_0.5` | C2Seg origin + KAN, scale 0.5 |
| `c2seg_nokan` | `qata_c2seg_mlp` | C2Seg origin + MLP K-Gate |
| `qata_nokan_deepsup` | `qata_c2seg_mlp_deepsup` | C2Seg + MLP + DeepSup |

### MosMedData+ Dataset

| Old Tag | New Tag | Description |
|---------|---------|-------------|
| `mosmed_nokan` | `mosmed_c2seg_mlp` | C2Seg origin + MLP K-Gate |
| `mosmed_nokan_deepsup` | `mosmed_c2seg_mlp_deepsup` | C2Seg + MLP + DeepSup |
| `mosmed_kan_deepsup` | `mosmed_c2seg_kan_deepsup` | C2Seg + KAN + DeepSup |
| `mosmed_full` | `mosmed_c2seg_full` | C2Seg full sync (latest) |

### Not Renamed (temp/debug runs)

- `mosmed_full_accum2` - debug run
- `mosmed_kan_deepsup_einsum_accum2` - debug run
- `qata_full` - debug run

---

## Server Rename Commands

Run on server to rename checkpoints:

```bash
cd /dscilab_hvch261_tuanminh/workspace/code/c2seg/output/checkpoints

# QaTa
mv efficient_kan_v1 qata_custom_kan
mv simple_kan_v1 qata_custom_kan_lite
mv ukan_0.5 qata_ukan_0.5
mv mlp_1.0 qata_custom_mlp
mv c2seg_kan qata_c2seg_kan
mv c2seg_kan_0.5 qata_c2seg_kan_0.5
mv c2seg_nokan qata_c2seg_mlp
mv qata_nokan_deepsup qata_c2seg_mlp_deepsup

# MosMed
mv mosmed_nokan mosmed_c2seg_mlp
mv mosmed_nokan_deepsup mosmed_c2seg_mlp_deepsup
mv mosmed_kan_deepsup mosmed_c2seg_kan_deepsup
mv mosmed_full mosmed_c2seg_full

# Similarly for logs
cd ../logs
# (same commands)
```

---

## Experiment Timeline

1. **Phase 1: Custom Implementation**
   - `qata_custom_kan` - B-spline KAN theo paper
   - `qata_custom_kan_lite` - Bỏ base spline giảm OOM
   - `qata_ukan_0.5` - Thử U-KAN block với scale 0.5
   - `qata_custom_mlp` - Ablation: full MLP thay KAN

2. **Phase 2: C2Seg Origin Code**
   - `qata_c2seg_kan` - Sync với code gốc, KAN K-Gate
   - `qata_c2seg_kan_0.5` - Scale 0.5 version
   - `qata_c2seg_mlp` - MLP K-Gate (better than KAN!)
   - `qata_c2seg_mlp_deepsup` - +DeepSup → **Best QaTa**

3. **Phase 3: MosMedData+**
   - `mosmed_c2seg_mlp` - MLP baseline
   - `mosmed_c2seg_mlp_deepsup` - +DeepSup
   - `mosmed_c2seg_kan_deepsup` - KAN wins on CT!
   - `mosmed_c2seg_full` - Full sync → **Best MosMed**

---

## Best Results

| Dataset | Tag | Dice | mIoU | HD95 | ASSD |
|---------|-----|------|------|------|------|
| QaTa-COV19 | `qata_c2seg_mlp_deepsup` | **84.00%** | **75.23%** | **15.53** | **3.03** |
| QaTa-COV19 | `qata_full` | 83.64% | 74.77% | - | - |
| MosMedData+ | `mosmed_c2seg_full` | **75.12%** | TBD | TBD | TBD |

---

## Comparison with SOTA Methods

### QaTa-COV19 Dataset

| Method | Params (M) | Text | Dice (%) | mIoU (%) | HD95 | ASSD |
|--------|------------|------|----------|----------|------|------|
| U-Net | 14.8 | - | 79.02 | 69.46 | - | - |
| nnUNet | 19.1 | - | 80.42 | 70.81 | - | - |
| TransUNet | 105.0 | - | 78.63 | 69.13 | - | - |
| Swin-Unet | 82.3 | - | 78.07 | 68.34 | - | - |
| SegFormer | 84.7 | - | 78.41 | 68.83 | - | - |
| ConVIRT | 35.2 | Y | 79.72 | 70.58 | - | - |
| TGANet | 19.8 | Y | 79.87 | 70.75 | - | - |
| CLIP | 87.0 | Y | 79.81 | 70.66 | - | - |
| GLORIA | 45.6 | Y | 79.94 | 70.68 | - | - |
| LViT | 29.7 | Y | 83.66 | 75.11 | - | - |
| LAVT | 118.6 | Y | 79.28 | 69.89 | - | - |
| DenseCLIP | 105.3 | Y | 79.58 | 70.37 | - | - |
| RefSegformer (Wu et al., 2024) | 193.0 | Y | 81.65 | 69.71 | 20.22 | 5.29 |
| **MedLangViT (Wang et al., 2025b)** | 27.7 | Y | 84.27 | 75.93 | 14.51 | 3.97 |
| **ARSeg (Wang et al., 2026)** | 30.1 | Y | 84.09 | 72.64 | 19.90 | 5.24 |
| **C2Seg (Ours)** | 18.92 | Y | **85.25** | **76.97** | **12.71** | **3.38** |

### MosMedData+ Dataset

| Method | Params (M) | Text | Dice (%) | mIoU (%) | HD95 | ASSD |
|--------|------------|------|----------|----------|------|------|
| U-Net | 14.8 | - | 64.60 | 50.73 | - | - |
| nnUNet | 19.1 | - | 72.59 | 60.36 | - | - |
| TransUNet | 105.0 | - | 71.24 | 58.44 | - | - |
| Swin-Unet | 82.3 | - | 63.29 | 50.19 | - | - |
| SegFormer | 84.7 | - | 65.05 | 54.34 | - | - |
| ConVIRT | 35.2 | Y | 72.06 | 59.73 | - | - |
| TGANet | 19.8 | Y | 71.81 | 59.28 | - | - |
| CLIP | 87.0 | Y | 71.97 | 59.64 | - | - |
| GLORIA | 45.6 | Y | 72.42 | 60.18 | - | - |
| LViT | 29.7 | Y | 74.57 | 61.33 | - | - |
| LAVT | 118.6 | Y | 73.29 | 60.41 | - | - |
| DenseCLIP | 105.3 | Y | 71.62 | 58.95 | - | - |
| RefSegformer (Wu et al., 2024) | 193.0 | Y | 70.25 | 57.51 | 19.70 | 4.78 |
| **MedLangViT (Wang et al., 2025b)** | 27.7 | Y | 75.95 | 63.17 | 18.29 | 4.12 |
| **ARSeg (Wang et al., 2026)** | 30.1 | Y | 73.24 | 59.82 | 31.88 | 7.65 |
| **C2Seg (Ours)** | 18.92 | Y | **77.81** | **65.17** | **15.02** | **3.76** |

### Key Observations

1. **C2Seg achieves SOTA** on both datasets with significantly fewer parameters (18.92M vs 27.7M MedLangViT)
2. **Text guidance** consistently improves performance over image-only methods
3. **C2Seg vs MedLangViT**: +0.98% Dice on QaTa, +1.86% Dice on MosMed with 31% fewer params
4. **C2Seg vs ARSeg**: +1.16% Dice on QaTa, +4.57% Dice on MosMed

### References

- MedLangViT: Wang et al., "MedLangViT: A Language-Vision Network for Medical Image Segmentation", Electronics 2025
- ARSeg: Wang et al., 2026
- LViT: Li et al., "LViT: Language meets Vision Transformer in Medical Image Segmentation", IEEE TMI 2023

---

*Last updated: October 2026*
