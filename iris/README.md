# IRIS - Medical Image Segmentation

Text-guided medical image segmentation with MLP-only baseline (no KAN).

## Summary

| Dataset | Our Best | C2Seg Paper | MedLangViT | Model |
|---------|----------|-------------|------------|-------|
| QaTa-COV19 | **84.25%** Dice | 85.25% | 84.27% | `qata_nopretrain_cnn` |
| MosMedData+ | **75.45%** Dice | 77.81% | 75.95% | `mosmed_pretrain_mix` |

Note: All results use **Macro Dice** (per-sample average) for fair comparison with published papers.

## Experiment Results

### QaTa-COV19 Dataset (Chest X-ray)

| Tag | Encoder | Pretrain | Test Dice | Test IoU |
|-----|---------|----------|-----------|----------|
| **`nopretrain_cnn`** | CNN | No | **84.25%** | **75.61%** |
| `nopretrain_mix` | Mix | No | 84.16% | 75.53% |
| `pretrain_mix` | Mix | Yes | 84.00% | 75.20% |
| `pretrain_cnn` | CNN | Yes | 83.72% | 74.85% |
| MedLangViT | - | - | 84.27% | 75.93% |
| ARSeg | - | - | 84.09% | 72.64% |
| **C2Seg (Paper)** | - | - | **85.25%** | **76.97%** |

### MosMedData+ Dataset (CT Scan)

| Tag | Encoder | Pretrain | Test Dice | Test IoU |
|-----|---------|----------|-----------|----------|
| **`pretrain_mix`** | Mix | Yes | **75.45%** | **62.33%** |
| `nopretrain_mix` | Mix | No | 73.08% | 59.29% |
| `nopretrain_cnn` | CNN | No | 73.03% | 59.30% |
| `pretrain_cnn` | CNN | Yes | 72.92% | 59.45% |
| MedLangViT | - | - | 75.95% | 63.17% |
| ARSeg | - | - | 73.24% | 59.82% |
| **C2Seg (Paper)** | - | - | **77.81%** | **65.17%** |

### Key Findings

**QaTa-COV19:**
- **Pretrain hurts slightly**: Without pretrain is better (84.25% vs 83.72%)
- **CNN and Mix similar**: Both achieve ~84%
- **Best config**: CNN + No Pretrain (Dice: 84.25%, IoU: 75.61%)
- **Competitive with SOTA**: Close to C2Seg paper (85.25%) and MedLangViT (84.27%)

**MosMedData+:**
- **Pretrain helps for Mix**: 75.45% vs 73.08%
- **Mix encoder better**: 75.45% vs 72.92% with pretrain
- **Best config**: Mix + Pretrain (Dice: 75.45%, IoU: 62.33%)
- **Competitive with baselines**: Close to MedLangViT (75.95%)

**General:**
- Simple MLP with expansion=4 achieves competitive results
- Dataset characteristics matter: different optimal configs for different datasets
- CaCL pretrain helps MosMed (CT) but slightly hurts QaTa (X-ray)

## Ablation Study: Text Encoder and Fusion Module

We conducted an ablation study to evaluate different text encoders and fusion modules while keeping MedKLIP visual encoder fixed.

### Configurations

| Component | Options | Description |
|-----------|---------|-------------|
| Visual Encoder | MedKLIP | Pre-trained medical vision encoder (fixed) |
| Text Encoder | CLIP, MedKLIP | CLIP (~106M params) vs MedKLIP/PubMedBERT (~153M params) |
| Fusion Module | BCAM, BiFMD | Bidirectional Cross-Attention vs Bidirectional Feature Mixing Decoder |

### QaTa-COV19 Results

| Text Encoder | Fusion | Dice (%) | IoU (%) | Params |
|--------------|--------|----------|---------|--------|
| CLIP | BCAM | **81.59** | **72.06** | ~106M |
| MedKLIP | BCAM | 80.78 | 71.11 | ~153M |
| CLIP | BiFMD | 80.28 | 70.41 | ~106M |
| MedKLIP | BiFMD | 79.07 | 69.39 | ~153M |

### Ablation Analysis

**BCAM vs BiFMD:**
- BCAM outperforms BiFMD consistently across both text encoders
- With CLIP text: BCAM +1.31% Dice (81.59% vs 80.28%)
- With MedKLIP text: BCAM +1.71% Dice (80.78% vs 79.07%)
- BCAM's bidirectional cross-attention provides better feature fusion

**CLIP vs MedKLIP Text Encoder:**
- CLIP text encoder outperforms MedKLIP text encoder
- With BCAM: CLIP +0.81% Dice (81.59% vs 80.78%)
- With BiFMD: CLIP +1.21% Dice (80.28% vs 79.07%)
- Despite MedKLIP being medical-domain pretrained, CLIP generalizes better

**Efficiency:**
- CLIP text encoder: ~106M parameters
- MedKLIP text encoder: ~153M parameters (+47M)
- CLIP achieves better results with fewer parameters

**Best Configuration:**
- **MedKLIP Visual + CLIP Text + BCAM Fusion** = 81.59% Dice
- Lighter and more effective than full MedKLIP configuration

### Running Ablation

```bash
# Run all 4 ablation experiments
./scripts/run_ablation.sh

# Quick test (2 epochs)
./scripts/run_ablation_test.sh
```

### Ablation Config

See `configs/qata_ablation.yaml` and `configs/mosmed_ablation.yaml` for ablation configurations.

## Usage

```bash
# Full pipeline: pretrain -> train -> eval
python run.py --config configs/qata.yaml --tag pretrain_cnn

# Skip pretrain
python run.py --config configs/qata.yaml --tag nopretrain_cnn --no_pretrain

# Use Mix encoder
python run.py --config configs/qata.yaml --tag pretrain_mix --encoder_type mix

# Eval only (appends to existing log)
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

## Metrics

All results use **Macro Dice** (per-sample average):
```python
dice_per_sample = [dice(sample_i) for i in range(N)]
macro_dice = mean(dice_per_sample)
```

This is the standard approach used in medical imaging papers, ensuring fair comparison regardless of image sizes or batch configurations.

## References

- C2Seg: "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)
- MedLangViT: Wang et al., 2025
- ARSeg: Wang et al., 2026
- Datasets:
  - [QaTa-COV19](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset) - Chest X-ray COVID-19 segmentation
  - [MosMedData+](https://mosmed.ai/) - CT scan COVID-19 segmentation

---

Last updated: October 2026
