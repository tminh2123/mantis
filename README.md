# C2Seg - Medical Image Segmentation

Paper: "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)

## Summary

| Dataset | Paper Target | Our Best | Gap | Model |
|---------|--------------|----------|-----|-------|
| QaTa-COV19 | 85.25% Dice | 84.00% | -1.25% | `qata_c2seg_mlp_deepsup` |
| QaTa-COV19 | 74.13% mIoU | 75.23% | +1.10% | `qata_c2seg_mlp_deepsup` |
| MosMedData+ | 77.81% Dice | 75.12% | -2.69% | `mosmed_c2seg_full` |

---

## Experiment Results

### QaTa-COV19 Dataset (Chest X-ray)

| Tag | Description | Dice | mIoU | HD95 | ASSD |
|-----|-------------|------|------|------|------|
| Paper | C2Seg official target | 85.25% | 74.13% | - | - |
| `qata_custom_kan` | Custom impl, B-spline KAN | 80.37% | 70.83% | 21.30 | 4.30 |
| `qata_custom_kan_lite` | Custom impl, no base spline | 79.38% | 69.51% | 24.45 | 4.92 |
| `qata_ukan_0.5` | U-KAN block, scale 0.5 | 79.46% | 69.71% | 21.69 | 4.47 |
| `qata_custom_mlp` | Custom impl, full MLP | 69.94% | 57.70% | 46.36 | 8.97 |
| `qata_c2seg_kan` | C2Seg origin + KAN K-Gate | 78.87% | 69.45% | 28.68 | 6.15 |
| `qata_c2seg_kan_0.5` | C2Seg origin + KAN, scale 0.5 | 78.43% | 68.73% | 29.54 | 6.36 |
| `qata_c2seg_mlp` | C2Seg origin + MLP K-Gate | 83.81% | 75.02% | 28.68 | 6.15 |
| **`qata_c2seg_mlp_deepsup`** | C2Seg + MLP + DeepSup | **84.00%** | **75.23%** | **15.53** | **3.03** |

### MosMedData+ Dataset (CT Scan)

| Tag | Description | Dice | mIoU | HD95 | ASSD |
|-----|-------------|------|------|------|------|
| Paper | C2Seg official target | ~77.81% | - | - | - |
| `mosmed_c2seg_mlp` | C2Seg origin + MLP K-Gate | 72.59% | 58.97% | 18.44 | 3.43 |
| `mosmed_c2seg_mlp_deepsup` | C2Seg + MLP + DeepSup | 73.74% | 60.23% | 21.06 | 4.20 |
| `mosmed_c2seg_kan_deepsup` | C2Seg + KAN + DeepSup | 74.04% | 60.76% | 18.20 | 3.53 |
| `mosmed_c2seg_kan_deepsup_einsum` | C2Seg + KAN + DeepSup + einsum | 70.32% | 56.84% | 19.70 | 4.94 |
| **`mosmed_c2seg_full`** | C2Seg full sync (latest) | **75.12%** | TBD | TBD | TBD |

---

## Architecture

```
                          C2Seg Architecture
+------------------------------------------------------------------+
|                                                                   |
|  +----------+    +-----------------+    +------------------+      |
|  |  Input   |    |  Visual Encoder |    |      BCAM        |      |
|  |  Image   |--->|  (5 DownBlocks) |--->|  (Cross Attn +   |      |
|  |  224x224 |    |   Conv-based    |    |   K-Gate Fusion) |      |
|  +----------+    +-----------------+    +--------+---------+      |
|                                                  |                |
|  +----------+    +-----------------+             |                |
|  |  Text    |--->|  CLIP Encoder   |-------------+                |
|  |  Prompt  |    |   (Frozen)      |                              |
|  +----------+    +-----------------+                              |
|                                                                   |
|                          +------------------+                     |
|                          |     Decoder      |                     |
|                          |  (4 UpBlocks +   |                     |
|                          |  PixLevelModule) |                     |
|                          +--------+---------+                     |
|                                   |                               |
|                     +-------------+-------------+                 |
|                     v                           v                 |
|               +----------+              +---------------+         |
|               | Main Out |              |Deep Supervision|        |
|               | 224x224  |              |(out1-4 scales)|         |
|               +----------+              +---------------+         |
|                                                                   |
+------------------------------------------------------------------+
```

See detailed diagrams: [c2seg/docs/](c2seg/docs/index.html)

---

## Quick Start

### Training Commands

```bash
# QaTa-COV19 Dataset
# MLP + DeepSup (recommended)
python run.py --stage pretrain --config configs/qata.yaml --tag qata_c2seg_mlp_deepsup --no-kan
python run.py --stage train    --config configs/qata.yaml --tag qata_c2seg_mlp_deepsup --no-kan
python run.py --stage eval     --config configs/qata.yaml --tag qata_c2seg_mlp_deepsup --no-kan

# MosMedData+ Dataset
# KAN + DeepSup (recommended)
python run.py --stage pretrain --config configs/mosmed.yaml --tag mosmed_c2seg_full --accum-steps 2
python run.py --stage train    --config configs/mosmed.yaml --tag mosmed_c2seg_full --accum-steps 2
python run.py --stage eval     --config configs/mosmed.yaml --tag mosmed_c2seg_full
```

### Evaluation from Checkpoint

```bash
python run.py --stage eval --config configs/qata.yaml \
    --checkpoint output/checkpoints/qata_c2seg_mlp_deepsup/c2seg_best.pth
```

---

## Project Structure

```
c2seg/
├── models/
│   ├── c2seg.py          # Main model architecture
│   ├── kan.py            # KANLinear implementation
│   ├── losses.py         # Loss functions (CaCL, DeepSup)
│   └── unet.py           # U-Net baseline
├── datasets/
│   └── dataset.py        # Data loading
├── utils/
│   └── seg_metrics.py    # Dice, mIoU, HD95, ASSD
├── configs/
│   ├── qata.yaml         # QaTa-COV19 config
│   └── mosmed.yaml       # MosMedData+ config
├── docs/                 # Architecture diagrams
└── run.py                # Training/eval script
```

---

## References

- Paper: "Towards Text-Mask Consistency in Medical Image Segmentation" (C2Seg)
- Datasets:
  - [QaTa-COV19](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset) - Chest X-ray COVID-19 segmentation
  - [MosMedData+](https://mosmed.ai/) - CT scan COVID-19 segmentation

---

Last updated: October 2026
