#!/bin/bash
# IRIS Ablation Study - QaTa-COV19 Dataset
# Run all 4 experiments sequentially

echo "======================================"
echo "IRIS Ablation Study"
echo "======================================"

cd /dscilab_hvch261_tuanminh/workspace/code/iris

# Experiment 1: CNN + Pretrain
echo ""
echo "[1/4] Running: CNN encoder + Pretrain"
echo "======================================"
python run.py --config configs/qata_cnn.yaml --mode full

# Experiment 2: CNN + No Pretrain
echo ""
echo "[2/4] Running: CNN encoder + No Pretrain"
echo "======================================"
python run.py --config configs/qata_cnn_nopretrain.yaml --mode full

# Experiment 3: Mix + Pretrain
echo ""
echo "[3/4] Running: Mix encoder + Pretrain"
echo "======================================"
python run.py --config configs/qata_mix.yaml --mode full

# Experiment 4: Mix + No Pretrain
echo ""
echo "[4/4] Running: Mix encoder + No Pretrain"
echo "======================================"
python run.py --config configs/qata_mix_nopretrain.yaml --mode full

echo ""
echo "======================================"
echo "All experiments completed!"
echo "Check results in output/logs/"
echo "======================================"
