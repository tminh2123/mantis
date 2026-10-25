# IRIS - Medical Image Segmentation

IRIS (Image Segmentation) - A text-guided medical image segmentation framework with MLP-only baseline (no KAN).

## Features

- CLIP text encoder for text guidance
- MLP with expansion (Transformer FFN style: C -> 4C -> C)
- BCA Module (Bi-directional Cross Attention)
- Gate Module for feature fusion
- Deep supervision loss
- Two encoder types: CNN (pure conv) and Mix (CNN + MLP)

## Project Structure

```
iris/
├── config.py           # Configuration (Python defaults + YAML + CLI)
├── run.py              # Training/eval script
├── datasets/           # Dataset loading
├── models/             # Model architectures
│   ├── iris.py         # Main IRIS model
│   └── losses.py       # Loss functions (CaCL, DeepSupervision, DiceBCE)
├── utils/              # Utilities (metrics)
├── configs/            # Dataset configs (qata.yaml, mosmed.yaml)
└── output/             # Logs and checkpoints
```

## Ablation Study Results

### QaTa-COV19 Dataset

| Experiment | Encoder | Pretrain | Val Dice | Test Dice | Test IoU |
|------------|---------|----------|----------|-----------|----------|
| baseline_pretrain_cnn | CNN | Yes | 87.95% | **89.08%** | **80.66%** |
| baseline_pretrain_mix | Mix | Yes | 88.17% | 88.92% | 80.42% |
| baseline_nopretrain_cnn | CNN | No | 87.26% | 88.13% | 79.16% |
| baseline_nopretrain_mix | Mix | No | 87.29% | 88.00% | 78.94% |

### Key Findings

1. **Pretrain helps**: ~1% improvement (89.08% vs 88.13% for CNN encoder)
2. **CNN encoder slightly better**: 89.08% vs 88.92% with pretrain
3. **Best configuration**: CNN encoder + Pretrain (Dice: 89.08%, IoU: 80.66%)
4. **MLP simplicity works**: Simple MLP with expansion=4 achieves strong results without complex KAN

### Training Details

- Pretrain: 100 epochs, batch_size=64, CaCL loss
- Train: 300 epochs (early stopping patience=10), batch_size=32
- Optimizer: Adam with differential LR (0.1x for visual encoder)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10)

## Usage

```bash
# Full pipeline (pretrain + train + eval)
python run.py --config configs/qata.yaml --mode full

# Train only (skip pretrain)
python run.py --config configs/qata.yaml --mode full --no_pretrain

# Use Mix encoder
python run.py --config configs/qata.yaml --mode full --encoder_type mix

# Evaluation with specific checkpoint
python run.py --config configs/qata.yaml --mode eval --checkpoint path/to/checkpoint.pth

# Override hyperparameters
python run.py --config configs/qata.yaml --mode full --epochs 200 --batch_size 16 --lr 0.0005
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Dataset config (required) | - |
| `--mode` | pretrain, train, eval, full | full |
| `--encoder_type` | cnn or mix | cnn |
| `--no_pretrain` | Skip pretrain stage | False |
| `--epochs` | Training epochs | 300 |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--es_patience` | Early stopping patience | 10 |
| `--checkpoint` | Override eval checkpoint | - |
| `--output_dir` | Output directory | output |

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
