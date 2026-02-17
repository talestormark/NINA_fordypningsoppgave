#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp007_pool
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 007: Late-Fusion Temporal Pooling (T=7)
# ============================================================================
#
# Research Question: Does recurrence help with longer sequences, or does
#   simple pooling suffice at T=7?
#
# Configuration:
#   - Model: Late-fusion with shared encoder + mean pool + 1x1 conv
#   - Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
#   - Temporal sampling: annual (T=7, 2018-2024)
#   - Bottleneck fusion: Mean pool over T -> 1x1 Conv (2048 -> 512)
#   - Skip aggregation: max (same as LSTM-UNet)
#   - Parameters: ~30.1M
#   - 5-fold stratified cross-validation
#
# Comparison:
#   exp007 vs exp001: Pool vs LSTM at T=7
#   If LSTM wins: recurrence helps at T=7
#   If pool wins/ties: simple pooling suffices at T=7
#
# Usage:
#   sbatch train_exp007_late_fusion_pool.sh [FOLD]
#
# Example:
#   for fold in 0 1 2 3 4; do sbatch train_exp007_late_fusion_pool.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 007: Late-Fusion Pool (T=7)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: Shared encoder + mean pool bottleneck + 1x1 conv fusion"
echo "Temporal sampling: annual (T=7, 2018-2024)"
echo "Config: AdamW + cosine + LR=0.01"
echo "Batch size: 4"
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
OUTPUT_DIR="PART1_multi_temporal_experiments/outputs/experiments/exp007_late_fusion_pool_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name late_fusion_pool \
    --temporal-sampling annual \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --skip-aggregation max \
    --batch-size 4 \
    --image-size 64 \
    --num-workers 4 \
    --epochs 400 \
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
