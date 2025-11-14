#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_train
#SBATCH --time=0-04:00:00           # 4 hours for full training
#SBATCH --partition=GPUQ            # GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32GB                  # 32GB RAM
#SBATCH --cpus-per-task=4           # 4 CPUs for data loading
#SBATCH --output=outputs/slurm_outputs/%x_%j.log
#SBATCH --error=outputs/slurm_outputs/%x_%j.err

# Script to train baseline land-take detection models on HPC

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=========================================="

# Load modules
module --quiet purge
module load Python/3.11.3-GCCcore-12.3.0

# Verify environment
echo ""
echo "Environment:"
python3 --version
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Create output directory
mkdir -p outputs/slurm_outputs
mkdir -p outputs/training

# Default training configuration
MODEL_NAME=${1:-"early_fusion"}    # early_fusion, siam_diff, or siam_conc
EPOCHS=${2:-"200"}                  # Number of epochs
BATCH_SIZE=${3:-"4"}                # Batch size
ENCODER_NAME=${4:-"resnet50"}       # Encoder architecture (resnet50, efficientnet-b4, etc.)
SEED=${5:-"42"}                     # Random seed for reproducibility
WANDB=${6:-"true"}                  # Enable wandb logging (true/false)
OUTPUT_DIR="outputs/training/${MODEL_NAME}_${ENCODER_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

# Set wandb flag
if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="--wandb"
else
    WANDB_FLAG=""
fi

echo "Training configuration:"
echo "  Model: $MODEL_NAME"
echo "  Encoder: $ENCODER_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo "  Wandb: $WANDB"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run training
python3 scripts/modeling/train.py \
    --model-name $MODEL_NAME \
    --encoder-name $ENCODER_NAME \
    --encoder-weights imagenet \
    --batch-size $BATCH_SIZE \
    --image-size 512 \
    --num-workers 4 \
    --epochs $EPOCHS \
    --lr 0.01 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --loss focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
    --output-dir $OUTPUT_DIR \
    --seed $SEED \
    $WANDB_FLAG

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
