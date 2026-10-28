#!/bin/bash
# Fusion Tuning Test - 1 epoch only
# Quick test to verify setup works

set -e

# Path to MedKLIP pretrained weights (update this path on server)
MEDKLIP_WEIGHTS="../MedKLIP/checkpoints/MedKLIP_resnet50.pth"

# Base config
CONFIG="configs/qata.yaml"
MODE="train"

# Fixed settings
DROPOUT="0.1"
EPOCHS=1

echo "=============================================="
echo "Fusion Tuning - TEST (1 epoch)"
echo "num_heads x expansion: {4, 6, 8} x {4, 6, 8}"
echo "Total: 9 experiments x 1 epoch"
echo "=============================================="

# Grid search
HEADS=(4 6 8)
EXPANSIONS=(4 6 8)

count=1
for h in "${HEADS[@]}"; do
    for e in "${EXPANSIONS[@]}"; do
        echo ""
        echo "[${count}/9] heads=${h}, expansion=${e} (1 epoch)"
        python run.py --config $CONFIG --tag fusion_test_h${h}_e${e} \
            --mode $MODE \
            --visual_encoder resnet \
            --visual_pretrained $MEDKLIP_WEIGHTS \
            --text_encoder clinicalbert \
            --fusion_type bifmd \
            --adapter multiscale \
            --visual_lr_scale 0.01 \
            --num_heads $h \
            --expansion $e \
            --attn_dropout $DROPOUT \
            --epochs $EPOCHS \
            --no_pretrain
        ((count++))
    done
done

echo ""
echo "=============================================="
echo "Test completed! All 9 configs work."
echo "Run ./scripts/run_fusion_tuning.sh for full training"
echo "=============================================="
