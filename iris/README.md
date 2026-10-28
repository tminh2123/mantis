# IRIS - Medical Image Segmentation

Text-guided medical image segmentation with MLP-only baseline (no KAN).

## Summary

| Dataset | Our Best | C2Seg Paper | MedLangViT | Model |
|---------|----------|-------------|------------|-------|
| QaTa-COV19 | **84.25%** Dice | 85.25% | 84.27% | `qata_nopretrain_cnn` |
| MosMedData+ | **75.45%** Dice | 77.81% | 75.95% | `mosmed_pretrain_mix` |

Note: All results use **Macro Dice** (per-sample average) for fair comparison with published papers.

## Experiment Results

### Configuration Legend

| Column | Values | Description |
|--------|--------|-------------|
| Pretrain | Yes/No | CaCL contrastive learning stage |
| Visual | CNN, Mix, MedKLIP | Visual encoder type |
| Text | CLIP, MedKLIP | Text encoder (frozen) |
| Fusion | BCAM, BiFMD | Fusion module type |

### QaTa-COV19 Dataset (Chest X-ray)

| Tag | Pretrain | Visual | Text | Fusion | Dice (%) | IoU (%) |
|-----|----------|--------|------|--------|----------|---------|
| **`nopretrain_cnn`** | No | CNN | CLIP | BCAM | **84.25** | **75.61** |
| `nopretrain_mix` | No | Mix | CLIP | BCAM | 84.16 | 75.53 |
| `pretrain_mix` | Yes | Mix | CLIP | BCAM | 84.00 | 75.20 |
| `pretrain_cnn` | Yes | CNN | CLIP | BCAM | 83.72 | 74.85 |
| `baseline_medklip_clip_bcam` | No | MedKLIP | CLIP | BCAM | 81.59 | 72.06 |
| `baseline_medklip_medklip_bcam` | No | MedKLIP | MedKLIP | BCAM | 80.78 | 71.11 |
| `baseline_medklip_clip_bifmd` | No | MedKLIP | CLIP | BiFMD | 80.28 | 70.41 |
| `baseline_medklip_medklip_bifmd` | No | MedKLIP | MedKLIP | BiFMD | 79.07 | 69.39 |
| MedLangViT | - | - | - | - | 84.27 | 75.93 |
| ARSeg | - | - | - | - | 84.09 | 72.64 |
| **C2Seg (Paper)** | - | - | - | - | **85.25** | **76.97** |

### MosMedData+ Dataset (CT Scan)

| Tag | Pretrain | Visual | Text | Fusion | Dice (%) | IoU (%) |
|-----|----------|--------|------|--------|----------|---------|
| **`pretrain_mix`** | Yes | Mix | CLIP | BCAM | **75.45** | **62.33** |
| `nopretrain_mix` | No | Mix | CLIP | BCAM | 73.08 | 59.29 |
| `nopretrain_cnn` | No | CNN | CLIP | BCAM | 73.03 | 59.30 |
| `pretrain_cnn` | Yes | CNN | CLIP | BCAM | 72.92 | 59.45 |
| MedLangViT | - | - | - | - | 75.95 | 63.17 |
| ARSeg | - | - | - | - | 73.24 | 59.82 |
| **C2Seg (Paper)** | - | - | - | - | **77.81** | **65.17** |

### Key Findings

**Visual Encoder:**
- CNN/Mix (custom) outperform MedKLIP visual encoder on QaTa
- Best: CNN without pretrain (84.25%) vs MedKLIP+CLIP+BCAM (81.59%)

**Text Encoder:**
- CLIP text encoder outperforms MedKLIP text encoder
- With MedKLIP visual + BCAM: CLIP +0.81% Dice (81.59% vs 80.78%)

**Fusion Module:**
- BCAM outperforms BiFMD consistently (~1.3-1.7% Dice improvement)

**Pretrain (CaCL):**
- Hurts QaTa slightly (84.25% vs 83.72% for CNN)
- Helps MosMed significantly (75.45% vs 73.08% for Mix)

**Best Configurations:**
- QaTa: CNN + No Pretrain + CLIP + BCAM = **84.25%** Dice
- MosMed: Mix + Pretrain + CLIP + BCAM = **75.45%** Dice

### Ablation Study Runs (Server Logs)

| Log File | Visual | Text | Notes |
|----------|--------|------|-------|
| `qata_ablation_cnn_clinicalbert_18.log` | CNN (4-level) | ClinicalBERT | max_len=18, Dice=83.28% |
| `qata_ablation_cnn_clinicalbert_bcam.log` | CNN (4-level) | ClinicalBERT | max_len=77, Dice=82.62% |
| `qata_ablation_cnn_medklip_bcam.log` | CNN (4-level) | PubMedBERT | MedKLIP text encoder, Dice=82.34% |
| `qata_ablation_medklip_clinicalbert.log` | MedKLIP ResNet50 | ClinicalBERT | Both medical encoders |
| `qata_ablation_medklip_frozen.log` | MedKLIP ResNet50 (frozen) | CLIP | Freeze visual, Dice=79.98% |
| `qata_ablation_native_medklip_clip_bcam.log` | MedKLIP ResNet50 (native channels) | CLIP | Native ResNet channels (64/256/512/1024) thay vì project sang CNN scale (64/128/256/512), Dice=81.07% |
| `qata_adapter_bottleneck.log` | MedKLIP ResNet50 (frozen+adapter) | CLIP | Bottleneck adapter, Dice=81.59% |
| `qata_adapter_spatial.log` | MedKLIP ResNet50 (frozen+adapter) | CLIP | Spatial adapter |
| `qata_baseline_temp.log` | CNN (5-level) | CLIP | Baseline test run |
| `qata_clinicalbert_5level.log` | CNN (5-level) | ClinicalBERT | Full IRIS + BERT text, Dice=83.08% |

**Ablation Summary:**
- CLIP text encoder > ClinicalBERT > PubMedBERT
- max_token_len=18 > max_token_len=77 for short medical prompts
- CNN visual > MedKLIP ResNet50 (domain mismatch: MedKLIP trained on X-ray, QaTa is CT)
- Channel adapters help MedKLIP encoder performance

### Running Ablation

```bash
# Run all ablation experiments (MedKLIP visual)
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
| `--visual_encoder` | No | cnn | cnn or resnet (MedKLIP) |
| `--text_encoder` | No | clip | clip, pubmedbert, clinicalbert |
| `--fusion_type` | No | bcam | bcam or bifmd |
| `--adapter` | No | None | bottleneck, spatial, multiscale |
| `--freeze_visual` | No | False | Freeze visual encoder |
| `--visual_pretrained` | No | None | Path to MedKLIP weights |
| `--num_heads` | No | 8 | Attention heads |
| `--attn_dropout` | No | 0.0 | Attention dropout |
| `--expansion` | No | 4 | MLP expansion ratio |
| `--max_token_len` | No | 18 | Max text token length |
| `--no_pretrain` | No | False | Skip pretrain stage |
| `--checkpoint` | No | - | Override checkpoint for eval |
| `--output_dir` | No | output | Output directory |

### Example Commands

```bash
# Baseline: CNN + CLIP (best for QaTa)
python run.py --config configs/qata.yaml --tag baseline

# CNN + ClinicalBERT
python run.py --config configs/qata.yaml --tag cnn_clinical \
    --text_encoder clinicalbert

# ResNet (MedKLIP) + CLIP
python run.py --config configs/qata.yaml --tag resnet_clip \
    --visual_encoder resnet \
    --visual_pretrained /path/to/medklip.pth

# Frozen ResNet + Bottleneck Adapter
python run.py --config configs/qata.yaml --tag resnet_adapter \
    --visual_encoder resnet --adapter bottleneck --freeze_visual \
    --visual_pretrained /path/to/medklip.pth

# Tune fusion params
python run.py --config configs/qata.yaml --tag tuned \
    --num_heads 4 --attn_dropout 0.1 --expansion 2
```

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
