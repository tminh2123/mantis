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
| MosMedData+ | `mosmed_c2seg_full` | **75.12%** | TBD | TBD | TBD |

---

*Last updated: October 2026*
