#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp006_lf
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 006: Late-Fusion Concat (Multi-View Aggregation)
# ============================================================================
#
# Research Question: RQ0 - Does recurrence help beyond simple multi-view aggregation?
#
# Configuration:
#   - Model: Late-fusion with shared encoder + concat + 1x1 conv fusion
#   - Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
#   - Temporal sampling: bi_temporal (2018 Q2, 2024 Q3)
#   - Bottleneck fusion: Concat → 1×1 Conv (2*2048 → 512)
#   - Skip aggregation: max (same as LSTM-UNet)
#   - 5-fold stratified cross-validation
#
# Architecture Comparison:
#   | Component      | LSTM-UNet (exp003)      | Late-Fusion (exp006)     |
#   |----------------|-------------------------|--------------------------|
#   | Encoder        | Shared ResNet-50        | Shared ResNet-50         |
#   | Bottleneck     | ConvLSTM (recurrent)    | Concat + 1×1 (static)    |
#   | Skip Agg       | max over time           | max over time            |
#   | Decoder        | UnetDecoder             | UnetDecoder              |
#   | Params         | ~70M                    | ~27M                     |
#
# Hypothesis:
#   LSTM-UNet should outperform late-fusion concat because:
#   - Recurrent gates model temporal dependencies explicitly
#   - ConvLSTM captures change dynamics, not just feature differences
#
#   If exp006 matches exp003:
#   - Simple aggregation suffices for this task
#   - LSTM adds complexity without benefit
#
# Expected Results:
#   - IoU: 45-52% (close to but below exp003's 53.29%)
#   - Should outperform exp005 (has shared encoder advantage)
#
# Usage:
#   sbatch train_exp006_late_fusion.sh [FOLD]
#
#   FOLD: 0, 1, 2, 3, or 4 (for 5-fold CV)
#
# Example:
#   # Run all 5 folds
#   for fold in 0 1 2 3 4; do sbatch train_exp006_late_fusion.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 006: Late-Fusion Concat"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: Shared encoder + concat bottleneck + 1x1 conv fusion"
echo "Temporal sampling: bi_temporal (2018 Q2, 2024 Q3)"
echo "Config: AdamW + cosine + LR=0.01"
echo "Batch size: 8"
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
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/exp006_late_fusion_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name late_fusion_concat \
    --temporal-sampling bi_temporal \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --skip-aggregation max \
    --batch-size 8 \
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
