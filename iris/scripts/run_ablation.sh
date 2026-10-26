#!/bin/bash
# Full Ablation Study for IRIS
# 
# 8 experiments total (4 per dataset):
#   - Datasets: QaTa-COV19, MosMed
#   - Visual: MedKLIP (fixed)
#   - Text: CLIP vs MedKLIP
#   - Fusion: BCAM vs BiFMD

set -e

QATA_CONFIG="configs/qata_ablation.yaml"
MOSMED_CONFIG="configs/mosmed_ablation.yaml"

echo "=========================================="
echo "IRIS Full Ablation Study"
echo "=========================================="

# -------------------------------------------
# QaTa-COV19 Dataset
# -------------------------------------------
echo ""
echo "============ QaTa-COV19 ============"

echo "[1/8] QaTa: MedKLIP + CLIP + BCAM"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_clip_bcam \
    --visual_encoder medklip --text_encoder clip --fusion_type bcam \
    --mode train

echo "[2/8] QaTa: MedKLIP + MedKLIP + BCAM"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_medklip_bcam \
    --visual_encoder medklip --text_encoder medklip --fusion_type bcam \
    --mode train

echo "[3/8] QaTa: MedKLIP + CLIP + BiFMD"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_clip_bifmd \
    --visual_encoder medklip --text_encoder clip --fusion_type bifmd \
    --mode train

echo "[4/8] QaTa: MedKLIP + MedKLIP + BiFMD"
python run.py --config $QATA_CONFIG --tag qata_baseline_medklip_medklip_bifmd \
    --visual_encoder medklip --text_encoder medklip --fusion_type bifmd \
    --mode train

# -------------------------------------------
# MosMed Dataset
# -------------------------------------------
echo ""
echo "============ MosMed ============"

echo "[5/8] MosMed: MedKLIP + CLIP + BCAM"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_clip_bcam \
    --visual_encoder medklip --text_encoder clip --fusion_type bcam \
    --mode train

echo "[6/8] MosMed: MedKLIP + MedKLIP + BCAM"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_medklip_bcam \
    --visual_encoder medklip --text_encoder medklip --fusion_type bcam \
    --mode train

echo "[7/8] MosMed: MedKLIP + CLIP + BiFMD"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_clip_bifmd \
    --visual_encoder medklip --text_encoder clip --fusion_type bifmd \
    --mode train

echo "[8/8] MosMed: MedKLIP + MedKLIP + BiFMD"
python run.py --config $MOSMED_CONFIG --tag mosmed_baseline_medklip_medklip_bifmd \
    --visual_encoder medklip --text_encoder medklip --fusion_type bifmd \
    --mode train

echo ""
echo "=========================================="
echo "Training Complete! Running Evaluation..."
echo "=========================================="

# Evaluate all
for tag in qata_baseline_medklip_clip_bcam qata_baseline_medklip_medklip_bcam \
           qata_baseline_medklip_clip_bifmd qata_baseline_medklip_medklip_bifmd; do
    echo "Evaluating $tag..."
    python run.py --config $QATA_CONFIG --tag $tag --mode eval
done

for tag in mosmed_baseline_medklip_clip_bcam mosmed_baseline_medklip_medklip_bcam \
           mosmed_baseline_medklip_clip_bifmd mosmed_baseline_medklip_medklip_bifmd; do
    echo "Evaluating $tag..."
    python run.py --config $MOSMED_CONFIG --tag $tag --mode eval
done

echo ""
echo "All Done!"
