#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp005_ef
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 005: Early-Fusion U-Net (No Temporal Modeling)
# ============================================================================
#
# Research Question: RQ0 - Do we need temporal modeling at all?
#
# Configuration:
#   - Model: Plain U-Net with 18-channel input (9 bands × 2 timesteps stacked)
#   - Encoder: ResNet-50 (ImageNet pretrained)
#   - Temporal sampling: bi_temporal (2018 Q2, 2024 Q3)
#   - No ConvLSTM - just channel stacking as early fusion
#   - 5-fold stratified cross-validation
#
# Hypothesis:
#   LSTM-UNet (exp003, 53.29% IoU) should outperform early-fusion U-Net because:
#   - ConvLSTM captures temporal dynamics that channel stacking cannot
#   - Shared encoder learns better features than treating timesteps as extra channels
#
# Expected Results:
#   - IoU: 40-50% (lower than exp003's 53.29%)
#   - If close to exp003: temporal modeling may be unnecessary for this task
#
# Usage:
#   sbatch train_exp005_early_fusion.sh [FOLD]
#
#   FOLD: 0, 1, 2, 3, or 4 (for 5-fold CV)
#
# Example:
#   # Run all 5 folds
#   for fold in 0 1 2 3 4; do sbatch train_exp005_early_fusion.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 005: Early-Fusion U-Net"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: Plain U-Net with stacked bi-temporal input (18 channels)"
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
OUTPUT_DIR="PART1_multi_temporal_experiments/outputs/experiments/exp005_early_fusion_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name early_fusion_unet \
    --temporal-sampling bi_temporal \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
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
