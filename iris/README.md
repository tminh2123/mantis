# IRIS - Medical Image Segmentation

Text-guided medical image segmentation with MLP-only baseline (no KAN).

## Features

- CLIP text encoder for text guidance (frozen)
- MLP with expansion (Transformer FFN style: C -> 4C -> C)
- BCA Module (Bi-directional Cross Attention)
- Gate Module for feature fusion
- Deep supervision loss
- Two encoder types: CNN (pure conv) and Mix (CNN + MLP)

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
└── output/
    ├── logs/           # {dataset}_{tag}.log
    └── checkpoints/    # {dataset}_{tag}/best.pth
```

## Usage

```bash
# Full pipeline: pretrain -> train -> eval
python run.py --config configs/qata.yaml --tag pretrain_cnn

# Skip pretrain
python run.py --config configs/qata.yaml --tag nopretrain_cnn --no_pretrain

# Use Mix encoder
python run.py --config configs/qata.yaml --tag pretrain_mix --encoder_type mix

# Eval only (uses checkpoint from tag)
python run.py --config configs/qata.yaml --tag pretrain_cnn --mode eval

# Eval with specific checkpoint
python run.py --config configs/qata.yaml --tag test --mode eval --checkpoint path/to/best.pth
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

## Ablation Study Results

### QaTa-COV19 Dataset (Chest X-ray)

| Experiment | Encoder | Pretrain | Val Dice | Test Dice | Test IoU |
|------------|---------|----------|----------|-----------|----------|
| baseline_pretrain_cnn | CNN | Yes | 87.95% | **89.08%** | **80.66%** |
| baseline_pretrain_mix | Mix | Yes | 88.17% | 88.92% | 80.42% |
| baseline_nopretrain_cnn | CNN | No | 87.26% | 88.13% | 79.16% |
| baseline_nopretrain_mix | Mix | No | 87.29% | 88.00% | 78.94% |

### MosMedData+ Dataset (CT Scan)

| Experiment | Encoder | Pretrain | Val Dice | Test Dice | Test IoU |
|------------|---------|----------|----------|-----------|----------|
| baseline_nopretrain_mix | Mix | No | 74.80% | **75.08%** | **61.03%** |
| baseline_pretrain_mix | Mix | Yes | 73.99% | 72.07% | 57.85% |
| baseline_nopretrain_cnn | CNN | No | 74.22% | 71.97% | 57.42% |
| baseline_pretrain_cnn | CNN | Yes | 73.85% | 69.91% | 56.05% |

### Key Findings

**QaTa-COV19:**
1. **Pretrain helps**: ~1% improvement (89.08% vs 88.13% for CNN encoder)
2. **CNN encoder slightly better**: 89.08% vs 88.92% with pretrain
3. **Best config**: CNN + Pretrain (Dice: 89.08%, IoU: 80.66%)

**MosMedData+:**
1. **Pretrain hurts**: Without pretrain is better (75.08% vs 72.07% for Mix)
2. **Mix encoder better**: 75.08% vs 71.97% without pretrain
3. **Best config**: Mix + No Pretrain (Dice: 75.08%, IoU: 61.03%)

**General:**
- MLP simplicity works: Simple MLP with expansion=4 achieves strong results without complex KAN
- Dataset characteristics matter: Different optimal configs for different datasets

## Training Details

| Parameter | QaTa | MosMed |
|-----------|------|--------|
| Pretrain epochs | 100 | 100 |
| Train epochs | 300 | 300 |
| Batch size | 32 | 32 |
| Learning rate | 0.001 | 0.0003 |
| Early stopping | 20 epochs | 20 epochs |
| Optimizer | Adam (0.1x LR for visual encoder) | Adam |
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
