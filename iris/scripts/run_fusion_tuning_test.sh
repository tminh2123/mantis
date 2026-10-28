#!/bin/bash
# Fusion Tuning Test - 1 epoch only
# Quick test to verify setup works

set -e

# Path to MedKLIP pretrained weights
MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"

# Base config
CONFIG="configs/qata.yaml"
MODE="train"

# Fixed settings
DROPOUT="0.1"
EPOCHS=1

echo "=============================================="
echo "Fusion Tuning - TEST (1 epoch)"
echo "num_heads x expansion: {4, 8} x {4, 6}"
echo "Total: 4 experiments x 1 epoch"
echo "=============================================="

# Grid search
HEADS=(4 8)
EXPANSIONS=(4 6)

count=1
total=4
for h in "${HEADS[@]}"; do
    for e in "${EXPANSIONS[@]}"; do
        echo ""
        echo "[${count}/${total}] heads=${h}, expansion=${e} (1 epoch)"
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
echo "Test completed! All ${total} configs work."
echo "Run ./scripts/run_fusion_tuning.sh for full training"
echo "=============================================="
