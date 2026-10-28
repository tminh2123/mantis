#!/bin/bash
# Fusion Tuning Test - 1 epoch only

set -e

MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"
CONFIG="configs/qata.yaml"
MODE="train"
DROPOUT="0.1"
EPOCHS=1

echo "=============================================="
echo "Fusion Tuning - TEST (1 epoch)"
echo "num_heads x expansion: {4, 8} x {2, 4, 8}"
echo "Total: 6 experiments x 1 epoch"
echo "=============================================="

HEADS=(4 8)
EXPANSIONS=(2 4 8)

count=1
total=6
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
echo "=============================================="
