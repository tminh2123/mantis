#!/bin/bash
# Decoder Attention Tuning - Remaining experiments
# l1_h4, l1_h8 done with bs64
# Now run l2 and l3 with bs32

set -e

MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"
CONFIG="configs/qata.yaml"

# Fixed from best fusion tuning (h8_e4)
NUM_HEADS=8
EXPANSION=4
PATIENCE=30
BATCH_SIZE=32

echo "=============================================="
echo "Decoder Attention Tuning - Remaining"
echo "l2_h4, l2_h8, l3_h4, l3_h8 with batch_size=${BATCH_SIZE}"
echo "=============================================="

EXPERIMENTS=(
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
        --batch_size $BATCH_SIZE \
        --decoder_attention \
        --decoder_attention_levels $levels \
        --decoder_attention_heads $heads
    
    ((count++))
done

echo ""
echo "=== RESULTS SUMMARY ==="
for exp in "${EXPERIMENTS[@]}"; do
    read -r levels heads <<< "$exp"
    TAG="decoder_l${levels}_h${heads}"
    echo "--- ${TAG} ---"
    grep -E "Test.*Dice" output/logs/qata_${TAG}.log 2>/dev/null | tail -1 || echo "No results"
done
