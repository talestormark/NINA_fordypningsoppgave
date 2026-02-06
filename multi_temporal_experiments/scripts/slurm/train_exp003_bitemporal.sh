#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp003_bit
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 003: LSTM-UNet with Bi-Temporal Sampling (T=2)
# ============================================================================
#
# Research Question: RQ2 - Sentinel-2 baseline (minimal temporal information)
#
# Configuration:
#   - Temporal sampling: bi_temporal (2 time steps: 2018 Q2, 2024 Q3)
#   - Model: LSTM-UNet with ResNet-50 encoder
#   - Optimized hyperparameters from exp001:
#     - Optimizer: AdamW
#     - Scheduler: Cosine annealing
#     - Learning rate: 0.01
#     - LSTM hidden dim: 256
#   - Batch size: 8 (increased due to lower memory with T=2)
#   - 5-fold stratified cross-validation
#
# Purpose:
#   Establish Sentinel-2 baseline for comparison with multi-temporal approaches.
#   Answer: How much does temporal trajectory information (T>2) improve detection?
#
# Usage:
#   sbatch train_exp003_bitemporal.sh [FOLD]
#
#   FOLD: 0, 1, 2, 3, or 4 (for 5-fold CV)
#
# Example:
#   # Run all 5 folds
#   for fold in 0 1 2 3 4; do sbatch train_exp003_bitemporal.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 003: LSTM-UNet Bi-Temporal (T=2)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Temporal sampling: bi_temporal (2018 Q2, 2024 Q3)"
echo "Config: AdamW + cosine + LR=0.01 + LSTM dim=256"
echo "Batch size: 8 (optimized for T=2)"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Navigate to project root
cd /cluster/home/tmstorma/NINA_fordypningsoppgave

# Load environment
module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Output directory
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/exp003_bitemporal_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling bi_temporal \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 2 \
    --skip-aggregation max \
    --batch-size 8 \
    --image-size 64 \
    --num-workers 4 \
    --epochs 200 \
    --lr 0.01 \
    --optimizer adamw \
    --scheduler cosine \
    --weight-decay 5e-4 \
    --loss focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --output-dir $OUTPUT_DIR \
    --seed 42 \
    --fold $FOLD \
    --num-folds 5 \
    --wandb \
    --wandb-project landtake-multitemporal

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
