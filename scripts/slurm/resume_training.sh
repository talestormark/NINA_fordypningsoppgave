#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=landtake_resume
#SBATCH --time=0-04:00:00           # 4 hours for continued training
#SBATCH --partition=GPUQ            # GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32GB                  # 32GB RAM
#SBATCH --cpus-per-task=4           # 4 CPUs for data loading
#SBATCH --output=outputs/slurm_outputs/resume_%j.log
#SBATCH --error=outputs/slurm_outputs/resume_%j.err

# Script to resume training from a checkpoint
# Usage: sbatch scripts/slurm/resume_training.sh <checkpoint_path> <total_epochs>
# Example: sbatch scripts/slurm/resume_training.sh outputs/training/early_fusion_20251113_094440/best_model.pth 400

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

# Get checkpoint path and total epochs from arguments
CHECKPOINT_PATH=${1:?"Error: Checkpoint path is required"}
TOTAL_EPOCHS=${2:-"400"}  # Default to 400 epochs if not specified

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Extract model configuration from checkpoint directory
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
CONFIG_FILE="$CHECKPOINT_DIR/config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse config (extract key values)
MODEL_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['model_name'])")
ENCODER_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['encoder_name'])")
ENCODER_WEIGHTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['encoder_weights'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['batch_size'])")
IMAGE_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['image_size'])")
NUM_WORKERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['num_workers'])")
LR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['lr'])")
MOMENTUM=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['momentum'])")
WEIGHT_DECAY=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['weight_decay'])")
LOSS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['loss'])")
FOCAL_ALPHA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['focal_alpha'])")
FOCAL_GAMMA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['focal_gamma'])")
SEED=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['seed'])")

echo "Resuming training:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Model: $MODEL_NAME"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Output directory: $CHECKPOINT_DIR"
echo ""

# Resume training
python3 scripts/modeling/train.py \
    --model-name $MODEL_NAME \
    --encoder-name $ENCODER_NAME \
    --encoder-weights $ENCODER_WEIGHTS \
    --batch-size $BATCH_SIZE \
    --image-size $IMAGE_SIZE \
    --num-workers $NUM_WORKERS \
    --epochs $TOTAL_EPOCHS \
    --lr $LR \
    --momentum $MOMENTUM \
    --weight-decay $WEIGHT_DECAY \
    --loss $LOSS \
    --focal-alpha $FOCAL_ALPHA \
    --focal-gamma $FOCAL_GAMMA \
    --output-dir $CHECKPOINT_DIR \
    --seed $SEED \
    --resume $CHECKPOINT_PATH

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
