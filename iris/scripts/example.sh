#!/bin/bash
# IRIS - Example Training Script

# =============================================================================
# BASIC USAGE
# =============================================================================

# Minimal - just config and tag
python run.py --config configs/qata.yaml --tag baseline

# =============================================================================
# FULL EXAMPLE
# =============================================================================

MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"

python run.py \
    --config configs/qata.yaml \
    --tag my_experiment \
    --mode train \
    --visual resnet \
    --text clinicalbert \
    --fusion bifmd \
    --pretrained $MEDKLIP_WEIGHTS \
    --freeze \
    --num_heads 8 \
    --expansion 4 \
    --lr 0.001 \
    --epochs 300 \
    --patience 30 \
    --batch_size 32 \
    --output_dir output

# =============================================================================
# ARGUMENT REFERENCE
# =============================================================================
#
# CORE:
#   --config      Config file (required)
#   --tag         Experiment tag (required)
#   --mode        train | eval
#   --output_dir  Output directory
#   --checkpoint  Checkpoint for eval
#
# MODEL:
#   --visual      cnn | resnet
#   --text        clip | pubmedbert | clinicalbert
#   --fusion      bcam | bifmd
#   --pretrained  Pretrained visual weights path
#   --freeze      Freeze visual encoder
#
# FUSION:
#   --num_heads    Attention heads (default: 8)
#   --expansion    MLP expansion (default: 4)
#   --memory_size  Linear memory size (0=disabled, try 32/64/128)
#
# TRAINING:
#   --lr          Learning rate
#   --epochs      Training epochs
#   --patience    Early stopping patience
#   --batch_size  Batch size
#
# =============================================================================
# BEST CONFIGS (QaTa-COV19)
# =============================================================================
#
# 1. Best (84.61% Dice):
#    --visual resnet --text clinicalbert --fusion bifmd --num_heads 8 --expansion 4
#
# 2. Baseline (84.25% Dice):
#    --visual cnn --text clip --fusion bcam
#
# =============================================================================
# EXPERIMENTS - Linear Memory
# =============================================================================
#
# Test with memory_size 64:
# python run.py --config configs/qata.yaml --tag memory_m64 \
#   --visual resnet --text clinicalbert --fusion bifmd \
#   --pretrained $MEDKLIP_WEIGHTS --freeze \
#   --num_heads 8 --expansion 4 --memory_size 64

# =============================================================================
# ANALYSIS SCRIPTS
# =============================================================================
#
# All analysis outputs go to output/analysis/
#
# 1. Text pattern analysis (no model needed):
#    python scripts/analyze_text.py --data_dir $DATA_DIR/train
#    Output: output/analysis/text_patterns/
#
# 2. Per-text Dice analysis (requires model):
#    python scripts/analyze_dice.py --config configs/qata.yaml \
#      --checkpoint output/checkpoints/EXP_NAME/best.pth \
#      --visual resnet --text clinicalbert --fusion bifmd --num_heads 8 --expansion 4
#    Output: output/analysis/per_text_dice/EXP_NAME/
#
# 3. Laterality analysis (requires model):
#    python scripts/analyze_laterality.py --config configs/qata.yaml \
#      --checkpoint output/checkpoints/EXP_NAME/best.pth \
#      --visual resnet --text clinicalbert --fusion bifmd --num_heads 8 --expansion 4
#    Output: output/analysis/laterality/EXP_NAME/
#
# 4. Visualization (requires model):
#    python scripts/visualize.py --config configs/qata.yaml \
#      --checkpoint output/checkpoints/EXP_NAME/best.pth \
#      --visual resnet --text clinicalbert --fusion bifmd --num_heads 8 --expansion 4
#    Output: output/analysis/visualizations/EXP_NAME/
