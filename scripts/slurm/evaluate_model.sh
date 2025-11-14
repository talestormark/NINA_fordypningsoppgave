#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_eval
#SBATCH --time=0-00:30:00           # 30 minutes for evaluation
#SBATCH --partition=GPUQ            # GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=16GB                  # 16GB RAM
#SBATCH --cpus-per-task=4           # 4 CPUs for data loading
#SBATCH --output=outputs/slurm_outputs/%x_%j.log
#SBATCH --error=outputs/slurm_outputs/%x_%j.err

# Script to evaluate trained model on test set

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
mkdir -p outputs/evaluation

# Default configuration
CHECKPOINT=${1:-"outputs/training/siam_conc_resnet50_20251113_094511/best_model.pth"}
OUTPUT_DIR=${2:-"outputs/evaluation/siam_conc_resnet50_seed42"}
BATCH_SIZE=${3:-"1"}
WANDB_RUN_ID=${4:-""}  # Optional: wandb run ID to resume

# Set wandb flags
if [ -n "$WANDB_RUN_ID" ]; then
    WANDB_FLAGS="--wandb --wandb-run-id $WANDB_RUN_ID"
    echo "Evaluation configuration:"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Wandb run ID: $WANDB_RUN_ID (will resume training run)"
else
    WANDB_FLAGS=""
    echo "Evaluation configuration:"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Wandb: disabled"
fi
echo ""

# Run evaluation
python3 scripts/modeling/evaluate.py \
    --checkpoint $CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --num-workers 4 \
    --image-size 512 \
    --save-predictions \
    --visualize \
    --num-viz-samples 8 \
    $WANDB_FLAGS

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
