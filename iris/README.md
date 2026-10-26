# IRIS - Medical Image Segmentation

Text-guided medical image segmentation with MLP-only baseline (no KAN).

## Summary

| Dataset | Our Best | C2Seg Paper | MedLangViT | Model |
|---------|----------|-------------|------------|-------|
| QaTa-COV19 | **89.84%** Dice | 85.25% | 84.27% | `qata_nopretrain_cnn` |
| MosMedData+ | **76.51%** Dice | 77.81% | 75.95% | `mosmed_pretrain_mix` |

## Experiment Results

### QaTa-COV19 Dataset (Chest X-ray)

| Tag | Encoder | Pretrain | Val Dice | Test Dice | Test IoU |
|-----|---------|----------|----------|-----------|----------|
| **`nopretrain_cnn`** | CNN | No | 88.57% | **89.84%** | **81.74%** |
| `nopretrain_mix` | Mix | No | 88.53% | 89.68% | 81.48% |
| `pretrain_cnn` | CNN | Yes | 88.24% | 89.49% | 81.18% |
| `pretrain_mix` | Mix | Yes | 88.36% | 89.47% | 81.17% |
| MedLangViT | - | - | - | 84.27% | 75.93% |
| ARSeg | - | - | - | 84.09% | 72.64% |
| **C2Seg (Paper)** | - | - | - | **85.25%** | **76.97%** |

### MosMedData+ Dataset (CT Scan)

| Tag | Encoder | Pretrain | Val Dice | Test Dice | Test IoU |
|-----|---------|----------|----------|-----------|----------|
| **`pretrain_mix`** | Mix | Yes | 77.99% | **76.51%** | **62.51%** |
| `nopretrain_cnn` | CNN | No | 76.87% | 74.31% | 59.76% |
| `nopretrain_mix` | Mix | No | 76.82% | 74.22% | 59.73% |
| `pretrain_cnn` | CNN | Yes | 77.06% | 74.19% | 59.74% |
| MedLangViT | - | - | - | 75.95% | 63.17% |
| ARSeg | - | - | - | 73.24% | 59.82% |
| **C2Seg (Paper)** | - | - | - | **77.81%** | **65.17%** |

### Key Findings

**QaTa-COV19:**
- **Pretrain hurts slightly**: Without pretrain is better (89.84% vs 89.49%)
- **CNN and Mix similar**: Both achieve ~89.5-89.8%
- **Best config**: CNN + No Pretrain (Dice: 89.84%, IoU: 81.74%)
- **Beats all baselines**: +4.6% over C2Seg paper, +5.6% over MedLangViT

**MosMedData+:**
- **Pretrain helps for Mix**: 76.51% vs 74.22%
- **Mix encoder better**: 76.51% vs 74.19% with pretrain
- **Best config**: Mix + Pretrain (Dice: 76.51%, IoU: 62.51%)
- **Competitive with SOTA**: Close to C2Seg paper (77.81%)

**General:**
- Simple MLP with expansion=4 outperforms complex architectures
- Dataset characteristics matter: different optimal configs for different datasets

## Usage

```bash
# Full pipeline: pretrain -> train -> eval
python run.py --config configs/qata.yaml --tag pretrain_cnn

# Skip pretrain
python run.py --config configs/qata.yaml --tag nopretrain_cnn --no_pretrain

# Use Mix encoder
python run.py --config configs/qata.yaml --tag pretrain_mix --encoder_type mix

# Eval only
python run.py --config configs/qata.yaml --tag pretrain_cnn --mode eval
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | - | Config file |
| `--tag` | Yes | - | Experiment tag |
| `--mode` | No | full | pretrain, train, eval, full |
| `--encoder_type` | No | cnn | cnn or mix |
| `--no_pretrain` | No | False | Skip pretrain stage |
| `--checkpoint` | No | - | Override checkpoint for eval |
| `--output_dir` | No | output | Output directory |

## Output Structure

```
output/
├── logs/
│   ├── qata_pretrain_cnn.log
│   └── mosmed_nopretrain_mix.log
└── checkpoints/
    ├── qata_pretrain_cnn/
    │   ├── pretrain.pth
    │   └── best.pth
    └── mosmed_nopretrain_mix/
        └── best.pth
```

## Project Structure

```
iris/
├── run.py              # Training/eval script
├── configs/            # YAML configs (qata.yaml, mosmed.yaml)
├── datasets/           # Dataset loading
├── models/
│   ├── iris.py         # Main IRIS model
│   └── losses.py       # CaCL, DeepSupervision, DiceBCE
├── utils/              # Metrics, utilities
└── output/             # Logs, checkpoints
```

## Training Details

| Parameter | QaTa | MosMed |
|-----------|------|--------|
| Pretrain epochs | 100 | 100 |
| Train epochs | 300 | 300 |
| Batch size | 32 | 32 |
| Learning rate | 0.001 | 0.0003 |
| Early stopping | 20 epochs | 20 epochs |
| Optimizer | Adam (0.1x LR for encoder) | Adam |
| Scheduler | CosineAnnealingWarmRestarts | CosineAnnealingWarmRestarts |

## Model Architecture

### Encoder Types

- **CNN**: Pure convolutional DownBlocks (Conv -> BN -> ReLU -> MaxPool)
- **Mix**: DownBlocks with MLP residual (CNN output + MLP(CNN output))

### MLP Design

Transformer FFN style with expansion factor:
```
Input (C) -> Linear (C -> 4C) -> GELU -> Dropout -> Linear (4C -> C) -> Output
```

### Training Stages

1. **Stage I (Pretrain)**: Category-aware Contrastive Learning (CaCL)
   - Learns visual-language alignment
   - Only visual encoder + projections trained (CLIP frozen)

2. **Stage II (Train)**: Segmentation with Deep Supervision
   - Multi-scale auxiliary outputs
   - Asymmetric loss weighting for class imbalance

## References

- C2Seg: "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)
- MedLangViT: Wang et al., 2025
- ARSeg: Wang et al., 2026
- Datasets:
  - [QaTa-COV19](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset) - Chest X-ray COVID-19 segmentation
  - [MosMedData+](https://mosmed.ai/) - CT scan COVID-19 segmentation

---

Last updated: October 2026
