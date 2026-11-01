#!/bin/bash
# IRIS - Example Training Script
# Shows all available CLI arguments

# =============================================================================
# BASIC USAGE
# =============================================================================

# Minimal - just config and tag
python run.py --config configs/qata.yaml --tag baseline

# =============================================================================
# FULL EXAMPLE WITH ALL ARGS
# =============================================================================

# Path to pretrained weights (optional)
MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"

python run.py \
    --config configs/qata.yaml \
    --tag my_experiment \
    --mode train \
    \
    `# === Encoder Selection ===` \
    --visual_encoder resnet \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --freeze_visual \
    \
    `# === Fusion Tuning ===` \
    --num_heads 8 \
    --expansion 4 \
    --attn_dropout 0.1 \
    \
    `# === Training Options ===` \
    --lr 0.001 \
    --visual_lr_scale 0.01 \
    --epochs 300 \
    --patience 30 \
    --batch_size 32 \
    --max_token_len 18 \
    \
    `# === Output ===` \
    --output_dir output

# =============================================================================
# ARGUMENT REFERENCE
# =============================================================================
#
# CORE:
#   --config           Config file path (required)
#   --tag              Experiment tag (required)
#   --mode             train (default) | eval
#   --output_dir       Output directory (default: output)
#   --checkpoint       Override checkpoint for eval
#
# ENCODER SELECTION:
#   --visual_encoder   cnn (default) | resnet
#   --text_encoder     clip (default) | pubmedbert | clinicalbert
#   --fusion_type      bcam (default) | bifmd
#   --adapter          Enable MultiScale adapter (flag)
#   --visual_pretrained Path to pretrained visual weights (e.g., MedKLIP)
#   --freeze_visual    Freeze visual encoder weights
#
# FUSION TUNING:
#   --num_heads        Attention heads in fusion (default: 8)
#   --expansion        MLP expansion ratio (default: 4)
#   --attn_dropout     Attention dropout (default: 0.0)
#
# TRAINING:
#   --lr               Base learning rate (default: from config)
#   --visual_lr_scale  LR scale for visual encoder (default: 0.1)
#   --epochs           Training epochs (default: from config)
#   --patience         Early stopping patience (default: 20)
#   --batch_size       Batch size (default: from config)
#   --max_token_len    Max text token length (default: 18)
#
# =============================================================================
# BEST CONFIGS (QaTa-COV19)
# =============================================================================
#
# 1. Best overall (84.61% Dice):
#    --visual_encoder resnet --text_encoder clinicalbert --fusion_type bifmd \
#    --adapter --num_heads 8 --expansion 4
#
# 2. Simple baseline (84.25% Dice):
#    --visual_encoder cnn --text_encoder clip --fusion_type bcam
#
