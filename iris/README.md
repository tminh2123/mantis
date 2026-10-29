# IRIS - Medical Image Segmentation

Text-guided medical image segmentation with MLP-only baseline (no KAN).

## Summary

| Dataset | Our Best | C2Seg Paper | MedLangViT | Model |
|---------|----------|-------------|------------|-------|
| QaTa-COV19 | **88.17%** Dice | 85.25% | 84.27% | `qata_baseline` (CNN+CLIP) |
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

### Fusion Tuning Results (MedKLIP + ClinicalBERT + BiFMD + MultiScale)

Grid search over num_heads and expansion_ratio with:
- Visual: MedKLIP ResNet50 (pretrained, lr_scale=0.01)
- Text: ClinicalBERT
- Fusion: BiFMD with improvements (MLP projection, LayerNorm, learnable temperature)
- Adapter: MultiScale
- Patience: 30 epochs

| Config | num_heads | expansion | Val Best Dice | Test Dice | Test IoU |
|--------|-----------|-----------|---------------|-----------|----------|
| h4_e2 | 4 | 2 | 82.16% | 84.18% | 75.63% |
| h4_e4 | 4 | 4 | 82.65% | 84.28% | 75.67% |
| **h4_e8** | **4** | **8** | 82.53% | **84.29%** | **75.81%** |
| **h8_e2** | **8** | **2** | 82.07% | **84.29%** | 75.62% |
| h8_e4 | 8 | 4 | - | 83.57% | 74.58% |

**Observations:**
- Best configs: h4_e8 and h8_e2 both achieve 84.29% Test Dice
- Larger expansion (8) with fewer heads (4) performs well
- h8_e4 underperforms - potential overfitting with 8 heads + 4 expansion
- MedKLIP+ClinicalBERT setup achieves ~84.3% vs baseline CLIP 88.17%

### Ablation Study Runs (Server Logs)

| Log File | Visual | Text | Notes |
|----------|--------|------|-------|
| `qata_ablation_cnn_clinicalbert_18.log` | CNN (4-level) | ClinicalBERT | max_len=18, Dice=81.30% |
| `qata_ablation_cnn_clinicalbert_bcam.log` | CNN (4-level) | ClinicalBERT | max_len=77, Dice=81.01% |
| `qata_ablation_cnn_medklip_bcam.log` | CNN (4-level) | PubMedBERT | MedKLIP text encoder, Dice=80.91% |
| `qata_ablation_medklip_clinicalbert.log` | MedKLIP ResNet50 | ClinicalBERT | Both medical encoders, Dice=79.83% |
| `qata_ablation_medklip_frozen.log` | MedKLIP ResNet50 (frozen) | CLIP | Freeze visual, Dice=78.87% |
| `qata_ablation_native_medklip_clip_bcam.log` | MedKLIP ResNet50 (native channels) | CLIP | Native ResNet channels (64/256/512/1024), Dice=79.57% |
| `qata_adapter_bottleneck.log` | MedKLIP ResNet50 (frozen+adapter) | CLIP | Bottleneck adapter, Dice=77.98% |
| `qata_adapter_spatial.log` | MedKLIP ResNet50 (frozen+adapter) | CLIP | Spatial adapter, Dice=78.84% |
| `qata_baseline_temp.log` | CNN (5-level) | CLIP | Baseline test run |
| `qata_clinicalbert_5level.log` | CNN (5-level) | ClinicalBERT | Full IRIS + BERT text, Dice=81.43% |
| `qata_baseline_medklip.log` | MedKLIP ResNet50 | CLIP | MedKLIP visual baseline, Dice=86.13% |
| `qata_baseline.log` | CNN (5-level) | CLIP | **Best baseline, Dice=88.17%** |

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
| `--lr` | No | config | Base learning rate |
| `--visual_lr_scale` | No | 0.1 | Visual encoder LR scale (e.g., 0.01 for pretrained) |
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

### Text Encoder Options

| Encoder | Model | Output Dim | Best For |
|---------|-------|------------|----------|
| **CLIP** | openai/clip-vit-base-patch32 | 512 | Cross-modal alignment |
| **ClinicalBERT** | emilyalsentzer/Bio_ClinicalBERT | 768 | Medical domain knowledge |
| **PubMedBERT** | microsoft/BiomedNLP-PubMedBERT | 768 | Biomedical literature |

Note: MedKLIP uses ClinicalBERT for pretraining. When using MedKLIP visual encoder, ClinicalBERT text encoder provides native alignment.

### Fusion Modules

- **BCAM**: Bidirectional Cross Attention Module
- **BiFMD**: Bidirectional FiLM-Modulated Decoder (improved version)

### BiFMD Improvements

BiFMD includes several architectural improvements for fusion tuning:

| Improvement | Description |
|-------------|-------------|
| **MLP Text Projection** | 2-layer MLP (768->768->512) with LayerNorm and GELU instead of single linear, preserves semantic info from BERT encoders |
| **Pre-Attention LayerNorm** | LayerNorm on visual and text features before FiLM modulation, stabilizes training |
| **Learnable Temperature** | Learnable attention temperature instead of fixed 1/sqrt(d), allows model to adjust attention sharpness |

### Fusion Tuning Parameters

| Parameter | CLI Arg | Default | Description |
|-----------|---------|---------|-------------|
| **num_heads** | `--num_heads` | 8 | Number of attention heads in fusion |
| **attn_dropout** | `--attn_dropout` | 0.0 | Dropout on attention weights |
| **expansion** | `--expansion` | 4 | MLP expansion ratio in fusion |

```bash
# Example: Tune fusion with fewer heads and dropout
python run.py --config configs/qata.yaml --tag fusion_tuned \
    --fusion_type bifmd \
    --num_heads 4 \
    --attn_dropout 0.1 \
    --expansion 2
```

### Visual Adapters

For parameter-efficient fine-tuning of pretrained visual encoders:

| Adapter | Description | Params/Layer | Best For |
|---------|-------------|--------------|----------|
| **Bottleneck** | 1x1 conv down -> 1x1 conv up | ~20K | Lightweight |
| **Spatial** | 1x1 down -> 3x3 conv -> 1x1 up | ~50K | Spatial context |
| **MultiScale** | Parallel 1x1 + 3x3 branches | ~80K | Segmentation (recommended) |

MultiScale adapter is recommended for segmentation tasks due to its spatial awareness and compatibility with deep supervision loss.

### Differential Learning Rate

When using pretrained visual encoders (e.g., MedKLIP), different components use different learning rates:

| Component | LR Scale | Rationale |
|-----------|----------|-----------|
| Visual encoder | 0.01-0.1x | Preserve pretrained features |
| Visual adapters | 1x | New trainable module |
| Text projection | 1x | New trainable module |
| Fusion modules | 1x | New trainable module |
| Decoder | 1x | New trainable module |

```bash
# Example: MedKLIP with MultiScale adapter
python run.py --config configs/qata.yaml --tag fusion_tune \
    --visual_encoder resnet \
    --visual_pretrained /path/to/medklip.pth \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --freeze_visual \
    --visual_lr_scale 0.01
```

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
