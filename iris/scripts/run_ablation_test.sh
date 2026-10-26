#!/bin/bash
# Quick Test - Run 2 epochs to check for errors
# Tag postfix: _test
# Mode: full (train + eval)

set -e
export TOKENIZERS_PARALLELISM=false

QATA_CONFIG="configs/qata_ablation.yaml"
MOSMED_CONFIG="configs/mosmed_ablation.yaml"
EPOCHS=2

echo "=========================================="
echo "IRIS Ablation Test (2 epochs)"
echo "=========================================="

# -------------------------------------------
# QaTa-COV19 Quick Test
# -------------------------------------------
echo ""
echo "============ QaTa-COV19 Test ============"

echo "[1/8] QaTa: MedKLIP + CLIP + BCAM"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_clip_bcam_test \
    --visual_encoder medklip --text_encoder clip --fusion_type bcam \
    --epochs $EPOCHS --mode full

echo "[2/8] QaTa: MedKLIP + MedKLIP + BCAM"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_medklip_bcam_test \
    --visual_encoder medklip --text_encoder medklip --fusion_type bcam \
    --epochs $EPOCHS --mode full

echo "[3/8] QaTa: MedKLIP + CLIP + BiFMD"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_clip_bifmd_test \
    --visual_encoder medklip --text_encoder clip --fusion_type bifmd \
    --epochs $EPOCHS --mode full

echo "[4/8] QaTa: MedKLIP + MedKLIP + BiFMD"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_medklip_bifmd_test \
    --visual_encoder medklip --text_encoder medklip --fusion_type bifmd \
    --epochs $EPOCHS --mode full

# -------------------------------------------
# MosMed Quick Test
# -------------------------------------------
echo ""
echo "============ MosMed Test ============"

echo "[5/8] MosMed: MedKLIP + CLIP + BCAM"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_clip_bcam_test \
    --visual_encoder medklip --text_encoder clip --fusion_type bcam \
    --epochs $EPOCHS --mode full

echo "[6/8] MosMed: MedKLIP + MedKLIP + BCAM"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_medklip_bcam_test \
    --visual_encoder medklip --text_encoder medklip --fusion_type bcam \
    --epochs $EPOCHS --mode full

echo "[7/8] MosMed: MedKLIP + CLIP + BiFMD"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_clip_bifmd_test \
    --visual_encoder medklip --text_encoder clip --fusion_type bifmd \
    --epochs $EPOCHS --mode full

echo "[8/8] MosMed: MedKLIP + MedKLIP + BiFMD"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_medklip_bifmd_test \
    --visual_encoder medklip --text_encoder medklip --fusion_type bifmd \
    --epochs $EPOCHS --mode full

echo ""
echo "=========================================="
echo "Test Complete! All 8 experiments OK."
echo "=========================================="
