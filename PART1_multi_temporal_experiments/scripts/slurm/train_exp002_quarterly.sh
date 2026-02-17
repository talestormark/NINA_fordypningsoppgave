#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp002_qtr
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 002: LSTM-UNet with Quarterly Temporal Sampling (T=14)
# ============================================================================
#
# Research Question: RQ2 - Does higher temporal density improve performance?
#
# Configuration:
#   - Temporal sampling: quarterly (14 time steps: Q2+Q3 for each year 2018-2024)
#   - Model: LSTM-UNet with ResNet-50 encoder
#   - Optimized hyperparameters from exp001:
#     - Optimizer: AdamW
#     - Scheduler: Cosine annealing
#     - Learning rate: 0.01
#     - LSTM hidden dim: 256
#   - Batch size: 2 with accumulation_steps=2 → effective batch size = 4
#     (Same effective batch size as exp001 for fair comparison)
#   - 5-fold stratified cross-validation
#
# Usage:
#   sbatch train_exp002_quarterly.sh [FOLD]
#
#   FOLD: 0, 1, 2, 3, or 4 (for 5-fold CV)
#
# Example:
#   # Run all 5 folds
#   for fold in 0 1 2 3 4; do sbatch train_exp002_quarterly.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 002: LSTM-UNet Quarterly (T=14)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Temporal sampling: quarterly (14 time steps)"
echo "Config: AdamW + cosine + LR=0.01 + LSTM dim=256"
echo "Batch size: 2 x 2 accumulation = 4 effective"
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
OUTPUT_DIR="PART1_multi_temporal_experiments/outputs/experiments/exp002_quarterly_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training with gradient accumulation
# batch_size=2 * accumulation_steps=2 = effective_batch_size=4
python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling quarterly \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 2 \
    --skip-aggregation max \
    --batch-size 2 \
    --accumulation-steps 2 \
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
