#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp011_lite7
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 011: LSTM-7-lite (T=7, ~30.3M params)
# ============================================================================
#
# Purpose: Close the parameter gap at T=7, isolating capacity from recurrence.
#   Both T=7 comparisons (Pool-7 ~30.1M, Conv3D-7 ~32.9M vs LSTM-7 54.4M)
#   have a ~23M parameter gap. LSTM-7-lite reduces the ConvLSTM to 1 layer,
#   h=32 (~30.3M total), matching the baselines. This is the T=7 analogue
#   of LSTM-2-lite (exp009).
#
# Configuration:
#   - Model: LSTM-UNet with reduced ConvLSTM (1 layer, hidden_dim=32)
#   - Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
#   - Temporal sampling: annual (T=7, 2018-2024)
#   - ConvLSTM: 1 layer, hidden_dim=32 (vs 2 layers, 256 in exp001)
#   - ConvLSTM kernel size: 3x3
#   - Skip aggregation: max
#   - Parameters: ~30.3M (matches Pool-7 and Conv3D-7)
#   - NO early stopping (full 400 epochs)
#   - 5-fold stratified cross-validation
#
# Comparisons:
#   exp011 vs exp007 (Pool-7): LSTM-lite vs Pool at T=7, param-matched
#   exp011 vs exp008 (Conv3D-7): LSTM-lite vs Conv3D at T=7, param-matched
#   exp011 vs exp010 (LSTM-7 no ES): Lite vs full LSTM at T=7
#
# Usage:
#   sbatch train_exp011_lstm7_lite.sh [FOLD]
#
# Example:
#   for fold in 0 1 2 3 4; do sbatch train_exp011_lstm7_lite.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 011: LSTM-7-lite (T=7, ~30.3M params)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: LSTM-UNet with 1-layer ConvLSTM, hidden_dim=32"
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
OUTPUT_DIR="PART1_multi_temporal_experiments/outputs/experiments/exp011_lstm7_lite_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 32 \
    --lstm-num-layers 1 \
    --convlstm-kernel-size 3 \
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
