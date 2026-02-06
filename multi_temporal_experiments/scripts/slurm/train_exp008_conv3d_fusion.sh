#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp008_3d
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 008: 3D Conv Bottleneck Fusion (T=7)
# ============================================================================
#
# Research Question: Does learned spatiotemporal filtering (non-recurrent)
#   match ConvLSTM recurrence at T=7?
#
# Configuration:
#   - Model: 3D conv fusion with shared encoder
#   - Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
#   - Temporal sampling: annual (T=7, 2018-2024)
#   - Bottleneck fusion: Conv3d(2048,512,k=(3,1,1)) x2 + mean pool over T
#   - Skip aggregation: max (same as LSTM-UNet)
#   - Parameters: ~35M
#   - 5-fold stratified cross-validation
#
# Comparison:
#   exp008 vs exp001: 3D Conv vs LSTM at T=7
#   If LSTM wins: recurrence > learned 3D filtering
#   If 3D Conv wins/ties: 3D conv captures temporal patterns without recurrence
#
# Usage:
#   sbatch train_exp008_conv3d_fusion.sh [FOLD]
#
# Example:
#   for fold in 0 1 2 3 4; do sbatch train_exp008_conv3d_fusion.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 008: 3D Conv Fusion (T=7)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: Shared encoder + temporal 3D conv (3,1,1) + mean pool"
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
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/exp008_conv3d_fusion_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name conv3d_fusion \
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
