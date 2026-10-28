#!/bin/bash
# Fusion Tuning Experiments - Full Grid Search
# num_heads x expansion: {4, 6, 8} x {4, 6, 8} = 9 runs

set -e

# Path to MedKLIP pretrained weights (update this path on server)
MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"

# Base config
CONFIG="configs/qata.yaml"
MODE="train"

# Fixed settings
DROPOUT="0.1"

echo "=============================================="
echo "Fusion Tuning - Full Grid Search"
echo "num_heads x expansion: {4, 6, 8} x {4, 6, 8}"
echo "Total: 9 experiments"
echo "=============================================="

# Grid search
HEADS=(4 6 8)
EXPANSIONS=(4 6 8)

count=1
for h in "${HEADS[@]}"; do
    for e in "${EXPANSIONS[@]}"; do
        echo ""
        echo "[${count}/9] heads=${h}, expansion=${e}"
        python run.py --config $CONFIG --tag fusion_h${h}_e${e} \
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
            --no_pretrain
        ((count++))
    done
done

echo ""
echo "=============================================="
echo "All 9 experiments completed!"
echo "Check logs in output/logs/"
echo "=============================================="
