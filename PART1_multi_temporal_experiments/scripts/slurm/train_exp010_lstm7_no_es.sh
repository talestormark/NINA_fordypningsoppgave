#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp010_no_es
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 010: LSTM-7 without Early Stopping
# ============================================================================
#
# Purpose: Eliminate training-budget asymmetry confound in T=7 architecture
#   comparisons (RQ2c, RQ2d). LSTM-7 (exp001_v2) used early stopping
#   (patience 30, min 50 epochs, best checkpoint at epochs 52-109), while
#   Pool-7 and Conv3D-7 ran full 400 epochs. This makes the cosine-annealing
#   LR trajectory differ. Retraining LSTM-7 for 400 epochs without early
#   stopping creates a fully controlled comparison.
#
# Configuration:
#   - Model: LSTM-UNet (identical to exp001_v2)
#   - Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
#   - Temporal sampling: annual (T=7, 2018-2024)
#   - ConvLSTM: 2 layers, hidden_dim=256
#   - Skip aggregation: max
#   - Parameters: ~54.4M (same as exp001_v2)
#   - NO early stopping (full 400 epochs)
#   - 5-fold stratified cross-validation
#
# Usage:
#   sbatch train_exp010_lstm7_no_es.sh [FOLD]
#
# Example:
#   for fold in 0 1 2 3 4; do sbatch train_exp010_lstm7_no_es.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 010: LSTM-7 without Early Stopping"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: LSTM-UNet (2-layer ConvLSTM, h=256)"
echo "Temporal sampling: annual (T=7, 2018-2024)"
echo "Config: AdamW + cosine + LR=0.01, NO early stopping"
echo "Batch size: 4"
echo "Epochs: 400 (full run)"
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
OUTPUT_DIR="PART1_multi_temporal_experiments/outputs/experiments/exp010_lstm7_no_es_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training (identical to exp001_v2 but WITHOUT early stopping)
python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 2 \
    --skip-aggregation max \
    --batch-size 4 \
    --accumulation-steps 1 \
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
