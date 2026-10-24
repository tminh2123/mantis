# IRIS - Medical Image Segmentation

IRIS (Image Segmentation) - A text-guided medical image segmentation framework.

## Features

- Vision Transformer backbone (MedCLIP-based)
- CLIP text encoder for text guidance
- CBAM fusion module
- Deep supervision loss

## Project Structure

```
iris/
├── config.py           # Configuration
├── run.py              # Training/eval script
├── datasets/           # Dataset loading
├── models/             # Model architectures
│   ├── iris.py         # Main IRIS model
│   ├── backbone.py     # Visual encoders (ViT, MedCLIP)
│   ├── fusion.py       # Fusion modules (CBAM, Cross-Attention)
│   ├── decoder.py      # Decoder with deep supervision
│   └── losses.py       # Loss functions
├── utils/              # Utilities
├── configs/            # Dataset configs
└── output/             # Outputs
```

## Experiment Plan

| Order | Experiment | Description |
|-------|------------|-------------|
| 1 | Baseline | Current best from C2Seg (MLP + DeepSup) |
| 2 | Backbone | MedCLIP / ViT backbone |
| 3 | Fusion | CBAM fusion module |
| 4 | Pretrain | Pretrained vs scratch |

## Usage

```bash
# Training
python run.py --config configs/qata.yaml --mode train

# Evaluation
python run.py --config configs/qata.yaml --mode eval
```
