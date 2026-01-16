#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=lstm_unet_train
#SBATCH --time=0-12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# SLURM job script for LSTM-UNet multi-temporal training
#
# Usage:
#   sbatch train_lstm_unet.sh [model] [sampling] [encoder] [seed] [batch_size] [epochs] [lstm_hidden] [lstm_layers] [wandb]
#
# Examples:
#   sbatch train_lstm_unet.sh lstm_unet annual resnet50 42 4 200 512 2 true
#   sbatch train_lstm_unet.sh lstm_unet quarterly resnet50 42 2 200 512 2 true

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=========================================="

# Change to working directory
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from: $WORKDIR"

# Load modules and activate environment
echo ""
echo "Loading environment..."
module --quiet purge
module load Anaconda3/2024.02-1

echo "Activating conda environment..."
source activate masterthesis

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check GPU
echo ""
echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create output directories
mkdir -p multi_temporal_experiments/outputs/logs
mkdir -p multi_temporal_experiments/outputs/experiments

# Parse arguments with defaults
MODEL_NAME=${1:-"lstm_unet"}
TEMPORAL_SAMPLING=${2:-"annual"}
ENCODER_NAME=${3:-"resnet50"}
SEED=${4:-"42"}
BATCH_SIZE=${5:-"4"}
EPOCHS=${6:-"200"}
LSTM_HIDDEN_DIM=${7:-"512"}
LSTM_NUM_LAYERS=${8:-"2"}
WANDB=${9:-"true"}
IMAGE_SIZE=${10:-"64"}  # Default 64 for Sentinel-2 tiles (~66x92 pixels)
LEARNING_RATE=${11:-"0.001"}  # Default 0.001 (10x smaller than baseline for stability)
FOLD=${12:-"none"}  # Fold index for k-fold CV (0-4), or "none" for original split
NUM_FOLDS=${13:-"5"}  # Number of folds for k-fold CV (default: 5)

# Create experiment ID
# Format: exp00X_model_sampling_seedXXX_foldX (if using k-fold CV)
if [ "$FOLD" = "none" ]; then
    EXP_ID="exp001_${MODEL_NAME}_${TEMPORAL_SAMPLING}_seed${SEED}"
else
    EXP_ID="exp001_${MODEL_NAME}_${TEMPORAL_SAMPLING}_seed${SEED}_fold${FOLD}"
fi
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/${EXP_ID}"

# Set wandb flag
if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb"
else
    WANDB_FLAG=""
fi

# Set fold flags
if [ "$FOLD" = "none" ]; then
    FOLD_FLAGS=""
    CV_MODE="Original train/val split (no k-fold CV)"
else
    FOLD_FLAGS="--fold $FOLD --num-folds $NUM_FOLDS"
    CV_MODE="${NUM_FOLDS}-fold CV (fold ${FOLD})"
fi

echo ""
echo "=========================================="
echo "TRAINING CONFIGURATION"
echo "=========================================="
echo "Experiment ID: $EXP_ID"
echo "Model: $MODEL_NAME"
echo "Temporal sampling: $TEMPORAL_SAMPLING"
echo "Encoder: $ENCODER_NAME"
echo "LSTM hidden dim: $LSTM_HIDDEN_DIM"
echo "LSTM layers: $LSTM_NUM_LAYERS"
echo "Batch size: $BATCH_SIZE"
echo "Image size: $IMAGE_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Wandb: $WANDB"
echo "Cross-validation: $CV_MODE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run training
python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name $MODEL_NAME \
    --temporal-sampling $TEMPORAL_SAMPLING \
    --encoder-name $ENCODER_NAME \
    --encoder-weights imagenet \
    --lstm-hidden-dim $LSTM_HIDDEN_DIM \
    --lstm-num-layers $LSTM_NUM_LAYERS \
    --skip-aggregation max \
    --batch-size $BATCH_SIZE \
    --image-size $IMAGE_SIZE \
    --num-workers 4 \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --loss focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --output-dir $OUTPUT_DIR \
    --seed $SEED \
    --wandb-project landtake-multitemporal \
    $WANDB_FLAG \
    $FOLD_FLAGS

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
