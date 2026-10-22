# C2Seg - Medical Image Segmentation

Implementation of C2Seg for COVID-19 lung infection segmentation using CLIP-guided cross-modal learning.

## Architecture

- **Encoder**: U-Net style with KAN (Kolmogorov-Arnold Networks) gates
- **Cross-modal fusion**: BCAM (Bidirectional Cross-Attention Module)  
- **Language encoder**: Frozen CLIP text encoder
- **Deep supervision**: Multi-scale loss with asymmetric weights

## Experiment Results

### QaTa-COV19 Dataset

| Experiment | KAN | DeepSup | Dice (%) | mIoU (%) | HD95 | ASSD |
|------------|-----|---------|----------|----------|------|------|
| c2seg_nokan | ❌ | ❌ | 83.81 | 75.02 | 14.40 | 2.98 |
| **qata_nokan_deepsup** | ❌ | ✅ | **84.00** | **75.23** | **15.53** | **3.03** |
| c2seg_kan | ✅ | ❌ | 78.87 | 69.45 | 28.68 | 6.15 |
| c2seg_kan_0.5 | ✅ (0.5x) | ❌ | 78.43 | 68.73 | 29.54 | 6.36 |

### MosMedData+ Dataset

| Experiment | KAN | DeepSup | Dice (%) | mIoU (%) | HD95 | ASSD |
|------------|-----|---------|----------|----------|------|------|
| mosmed_nokan | ❌ | ❌ | 72.59 | 58.97 | 18.44 | 3.43 |
| mosmed_nokan_deepsup | ❌ | ✅ | 73.74 | 60.23 | 21.06 | 4.20 |
| **mosmed_kan_deepsup** | ✅ | ✅ | **74.04** | **60.76** | **18.20** | **3.53** |
| mosmed_kan_deepsup_einsum_accum2 | ✅ | ✅ | 70.32 | 56.84 | 19.70 | 4.94 |

### Ablation Studies (QaTa-COV19)

| Experiment | Description | Dice (%) | mIoU (%) | HD95 | ASSD |
|------------|-------------|----------|----------|------|------|
| mlp_1.0 | MLP baseline | 69.94 | 57.70 | 46.36 | 8.97 |
| ukan_0.5 | U-KAN (0.5x channels) | 79.46 | 69.71 | 21.69 | 4.47 |
| simple_kan_v1 | Simple KAN | 79.38 | 69.51 | 24.45 | 4.92 |
| efficient_kan_v1 | Efficient KAN | 80.37 | 70.83 | 21.30 | 4.30 |

## Key Findings

1. **Deep Supervision helps**: +0.19% Dice on QaTa-COV19, +1.15% on MosMedData+
2. **KAN gates hurt on QaTa-COV19**: -5% Dice compared to no-KAN baseline
3. **KAN gates help on MosMedData+**: +1.45% Dice with DeepSup
4. **Best configuration**:
   - QaTa-COV19: No KAN + Deep Supervision (84.00% Dice)
   - MosMedData+: KAN + Deep Supervision (74.04% Dice)

## Usage

### Pretrain (Contrastive Learning)
```bash
python run.py --stage pretrain --config configs/qata.yaml --tag qata_exp
```

### Train (Segmentation)
```bash
python run.py --stage train --config configs/qata.yaml --tag qata_exp --pretrained output/checkpoints/qata_exp/c2seg_pretrain.pth
```

### Evaluate
```bash
python run.py --stage eval --config configs/qata.yaml --tag qata_exp
```

### Options
- `--batch-size`: Override batch size
- `--accum-steps`: Gradient accumulation steps (auto-divides batch size)
- `--lr`: Override learning rate
- `--epochs`: Override number of epochs

## Configuration

See `configs/qata.yaml` and `configs/mosmed.yaml` for full configuration options.

Key parameters:
- `model.base_channel`: 64 (default)
- `model.kan_grid_size`: 5 (KAN B-spline intervals)
- `model.kan_spline_order`: 3 (cubic splines)
- `train.loss_weights`: [0.5, 0.2, 0.15, 0.1, 0.05] (deep supervision)

## Requirements

```
torch>=2.0
transformers
opencv-python-headless
numpy<2.0
scipy
pandas
openpyxl
tqdm
```

## Project Structure

```
c2seg/
├── configs/          # YAML configurations
├── datasets/         # Dataset loading
├── models/           # Model architecture
│   ├── c2seg.py      # Main model
│   ├── kan.py        # KAN implementation
│   ├── losses.py     # Loss functions
│   └── unet.py       # U-Net blocks
├── utils/            # Metrics and utilities
├── output/           # Checkpoints and logs
└── run.py            # Training script
```
