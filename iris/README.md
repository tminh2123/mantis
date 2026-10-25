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

### Training Details

- Pretrain: 100 epochs, batch_size=64, CaCL loss
- Train: 300 epochs (early stopping patience=10), batch_size=32
- Optimizer: Adam with differential LR (0.1x for visual encoder)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10)

## Usage

```bash
# Full pipeline (pretrain + train + eval)
python run.py --config configs/qata.yaml --tag baseline_cnn --mode full

# Different experiments with unique tags
python run.py --config configs/qata.yaml --tag attn_v1 --mode full
python run.py --config configs/qata.yaml --tag gate_mix --encoder_type mix --mode full
python run.py --config configs/qata.yaml --tag nopre_cnn --no_pretrain --mode full

# MosMedData+
python run.py --config configs/mosmed.yaml --tag baseline_cnn --mode full

# Evaluation with specific checkpoint
python run.py --config configs/qata.yaml --tag baseline_cnn --mode eval
python run.py --config configs/qata.yaml --tag baseline_cnn --mode eval --checkpoint path/to/checkpoint.pth

# Override hyperparameters
python run.py --config configs/qata.yaml --tag lr_test --epochs 200 --batch_size 16 --lr 0.0005
```

### Output Structure

```
output/
├── logs/
│   ├── qata_baseline_cnn/
│   │   ├── iris_pretrain.log
│   │   ├── iris_train.log
│   │   └── iris_eval.log
│   ├── qata_attn_v1/
│   └── mosmed_baseline_cnn/
└── checkpoints/
    ├── qata_baseline_cnn/
    │   ├── pretrain.pth
    │   ├── best.pth
    │   └── epoch10.pth
    ├── qata_attn_v1/
    └── mosmed_baseline_cnn/
```

### CLI Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--config` | Dataset config file | Yes |
| `--tag` | Experiment tag (unique identifier) | Yes |
| `--mode` | pretrain, train, eval, full | No (default: full) |
| `--encoder_type` | cnn or mix | No (default: cnn) |
| `--no_pretrain` | Skip pretrain stage | No |
| `--epochs` | Training epochs | No (default: 300) |
| `--batch_size` | Batch size | No (default: 32) |
| `--lr` | Learning rate | No (default: 0.001) |
| `--es_patience` | Early stopping patience | No (default: 10) |
| `--checkpoint` | Override eval checkpoint | No |
| `--output_dir` | Output directory | No (default: output) |

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
