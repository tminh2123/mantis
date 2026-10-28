#!/bin/bash
# Fusion Tuning Experiments
# Run on server with MedKLIP visual + ClinicalBERT text + BiFMD fusion
# Tune: num_heads (4, 8) x expansion (4, 8)

set -e

# Path to MedKLIP pretrained weights (update this path on server)
MEDKLIP_WEIGHTS="../MedKLIP/checkpoints/MedKLIP_resnet50.pth"

# Base config
CONFIG="configs/qata.yaml"
MODE="train"  # Skip pretrain, just train segmentation

# Common settings
DROPOUT="0.1"  # Fixed dropout for all experiments

echo "=============================================="
echo "Fusion Tuning Experiments"
echo "num_heads x expansion grid search"
echo "attn_dropout=${DROPOUT} (fixed)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Experiment 1: heads=4, expansion=4 (baseline)
# -----------------------------------------------------------------------------
echo ""
echo "[1/4] heads=4, expansion=4 (baseline)"
python run.py --config $CONFIG --tag fusion_h4_e4 \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --num_heads 4 \
    --expansion 4 \
    --attn_dropout $DROPOUT \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 2: heads=4, expansion=8
# -----------------------------------------------------------------------------
echo ""
echo "[2/4] heads=4, expansion=8"
python run.py --config $CONFIG --tag fusion_h4_e8 \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --num_heads 4 \
    --expansion 8 \
    --attn_dropout $DROPOUT \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 3: heads=8, expansion=4
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] heads=8, expansion=4"
python run.py --config $CONFIG --tag fusion_h8_e4 \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --num_heads 8 \
    --expansion 4 \
    --attn_dropout $DROPOUT \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 4: heads=8, expansion=8
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] heads=8, expansion=8"
python run.py --config $CONFIG --tag fusion_h8_e8 \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --num_heads 8 \
    --expansion 8 \
    --attn_dropout $DROPOUT \
    --no_pretrain

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Grid: num_heads={4,8} x expansion={4,8}"
echo "Check logs in output/logs/"
echo "=============================================="
