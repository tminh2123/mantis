#!/bin/bash
# Re-evaluate fusion tuning experiments with fixed metrics

set -e

# Path to MedKLIP pretrained weights
MEDKLIP_WEIGHTS="medklip_checkpoints/MedKLIP/checkpoint_final.pth"

# Base config
CONFIG="configs/qata.yaml"

echo "=============================================="
echo "Fusion Tuning - Re-Evaluation"
echo "=============================================="

# Grid search params - must match training
HEADS=(4 8)
EXPANSIONS=(2 4 8)

count=1
total=6
for h in "${HEADS[@]}"; do
    for e in "${EXPANSIONS[@]}"; do
        TAG="fusion_h${h}_e${e}"
        CKPT="output/checkpoints/qata_${TAG}/best.pth"
        
        if [ -f "$CKPT" ]; then
            echo ""
            echo "[${count}/${total}] Evaluating: ${TAG}"
            python run.py --config $CONFIG --tag $TAG \
                --mode eval \
                --visual_encoder resnet \
                --visual_pretrained $MEDKLIP_WEIGHTS \
                --text_encoder clinicalbert \
                --fusion_type bifmd \
                --adapter multiscale \
                --num_heads $h \
                --expansion $e \
                --no_pretrain
        else
            echo "[${count}/${total}] SKIP: ${TAG} - checkpoint not found"
        fi
        ((count++))
    done
done

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results appended to logs in output/logs/"
echo "=============================================="

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
grep "Test" output/logs/qata_fusion_*.log 2>/dev/null || echo "No results found"
