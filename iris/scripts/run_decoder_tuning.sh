#!/bin/bash
# Decoder Attention Tuning
# Base: h8_e4 (best fusion config: 84.61%)
# Tune: decoder_attention_levels (1,2,3) x decoder_attention_heads (4,8)

set -e

# Path to MedKLIP pretrained weights
MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"

# Base config
CONFIG="configs/qata.yaml"

# Fixed from best fusion tuning
NUM_HEADS=8
EXPANSION=4
PATIENCE=30

echo "=============================================="
echo "Decoder Attention Tuning"
echo "Base: h8_e4 (84.61% Dice)"
echo "=============================================="

# Grid: levels (1,2,3) x heads (4,8)
EXPERIMENTS=(
    "1 4"
    "1 8"
    "2 4"
    "2 8"
    "3 4"
    "3 8"
)

count=1
total=${#EXPERIMENTS[@]}
for exp in "${EXPERIMENTS[@]}"; do
    read -r levels heads <<< "$exp"
    TAG="decoder_l${levels}_h${heads}"
    
    echo ""
    echo "[${count}/${total}] Training: ${TAG}"
    echo "  decoder_attention_levels=${levels}, decoder_attention_heads=${heads}"
    
    python run.py --config $CONFIG --tag $TAG \
        --mode train \
        --visual_encoder resnet \
        --visual_pretrained $MEDKLIP_WEIGHTS \
        --text_encoder clinicalbert \
        --fusion_type bifmd \
        --adapter multiscale \
        --visual_lr_scale 0.01 \
        --num_heads $NUM_HEADS \
        --expansion $EXPANSION \
        --patience $PATIENCE \
        --decoder_attention \
        --decoder_attention_levels $levels \
        --decoder_attention_heads $heads
    
    ((count++))
done

echo ""
echo "=============================================="
echo "All ${total} experiments completed!"
echo "=============================================="

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for exp in "${EXPERIMENTS[@]}"; do
    read -r levels heads <<< "$exp"
    TAG="decoder_l${levels}_h${heads}"
    echo "--- ${TAG} ---"
    grep -E "Test.*Dice|Best Dice" output/logs/qata_${TAG}.log 2>/dev/null | tail -2 || echo "No results"
done
