#!/bin/bash
# Fusion Tuning Experiments
# Run on server with MedKLIP visual + ClinicalBERT text + BiFMD fusion

set -e

# Path to MedKLIP pretrained weights (update this path on server)
MEDKLIP_WEIGHTS="../MedKLIP/checkpoints/MedKLIP_resnet50.pth"

# Base config
CONFIG="configs/qata.yaml"
MODE="train"  # Skip pretrain, just train segmentation

echo "=============================================="
echo "Fusion Tuning Experiments"
echo "=============================================="

# -----------------------------------------------------------------------------
# Experiment 1: Baseline - MedKLIP + ClinicalBERT + BiFMD + MultiScale Adapter
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Baseline: BiFMD + MultiScale Adapter"
python run.py --config $CONFIG --tag fusion_baseline \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 2: Tune num_heads (2 vs default 4 for BiFMD)
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Tune: num_heads=2"
python run.py --config $CONFIG --tag fusion_heads2 \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --num_heads 2 \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 3: Tune attn_dropout (0.1)
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Tune: attn_dropout=0.1"
python run.py --config $CONFIG --tag fusion_dropout \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --attn_dropout 0.1 \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 4: Tune expansion (2 vs 4)
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Tune: expansion=2"
python run.py --config $CONFIG --tag fusion_expand2 \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --expansion 2 \
    --no_pretrain

# -----------------------------------------------------------------------------
# Experiment 5: Combined (heads=2, dropout=0.1, expansion=2)
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Combined: heads=2, dropout=0.1, expansion=2"
python run.py --config $CONFIG --tag fusion_combined \
    --mode $MODE \
    --visual_encoder resnet \
    --visual_pretrained $MEDKLIP_WEIGHTS \
    --text_encoder clinicalbert \
    --fusion_type bifmd \
    --adapter multiscale \
    --visual_lr_scale 0.01 \
    --num_heads 2 \
    --attn_dropout 0.1 \
    --expansion 2 \
    --no_pretrain

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Check logs in output/logs/"
echo "=============================================="
