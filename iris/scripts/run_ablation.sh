#!/bin/bash
# IRIS Ablation Study
# Usage: ./scripts/run_ablation.sh [qata|mosmed]

DATASET=${1:-qata}

# Auto-detect base path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IRIS_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo "IRIS Ablation - Dataset: $DATASET"
echo "Working dir: $IRIS_DIR"
echo "======================================"

cd "$IRIS_DIR"

# 1. CNN + Pretrain
echo "[1/4] baseline_pretrain_cnn"
python run.py --config configs/${DATASET}.yaml --mode full

# 2. CNN + No Pretrain
echo "[2/4] baseline_nopretrain_cnn"
python run.py --config configs/${DATASET}.yaml --mode full --no_pretrain

# 3. Mix + Pretrain
echo "[3/4] baseline_pretrain_mix"
python run.py --config configs/${DATASET}.yaml --mode full --encoder_type mix

# 4. Mix + No Pretrain
echo "[4/4] baseline_nopretrain_mix"
python run.py --config configs/${DATASET}.yaml --mode full --encoder_type mix --no_pretrain

echo "Done! Check output/logs/"
