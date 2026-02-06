#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp_v2
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Re-run All Experiments with Per-Fold Normalization (v2)
# ============================================================================
#
# Changes from v1:
#   - Normalization stats computed per fold from TRAINING samples only
#   - This avoids data leakage during cross-validation
#   - Early stopping with patience=30, min_epochs=50
#   - Precision/recall now printed in logs
#
# Usage:
#   sbatch train_all_experiments_v2.sh <EXPERIMENT> <FOLD>
#
#   EXPERIMENT: exp001 (annual T=7), exp002 (quarterly T=14), exp003 (bi-temporal T=2)
#   FOLD: 0, 1, 2, 3, or 4 (for 5-fold CV)
#
# Examples:
#   # Run single experiment/fold
#   sbatch train_all_experiments_v2.sh exp001 0
#
#   # Run all folds for exp001
#   for fold in 0 1 2 3 4; do sbatch train_all_experiments_v2.sh exp001 $fold; done
#
#   # Run all experiments, all folds
#   for exp in exp001 exp002 exp003; do
#     for fold in 0 1 2 3 4; do
#       sbatch train_all_experiments_v2.sh $exp $fold
#     done
#   done
#
# ============================================================================

EXPERIMENT=${1:-"exp001"}
FOLD=${2:-"0"}

# Set experiment-specific parameters
case $EXPERIMENT in
    "exp001")
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        ;;
    "exp002")
        TEMPORAL_SAMPLING="quarterly"
        TIME_STEPS=14
        BATCH_SIZE=2
        ACCUM_STEPS=2
        ;;
    "exp003")
        TEMPORAL_SAMPLING="bi_temporal"
        TIME_STEPS=2
        BATCH_SIZE=8
        ACCUM_STEPS=1
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Valid options: exp001, exp002, exp003"
        exit 1
        ;;
esac

EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUM_STEPS))

echo "=========================================="
echo "EXPERIMENT: $EXPERIMENT (v2 - per-fold normalization)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Temporal sampling: $TEMPORAL_SAMPLING (T=$TIME_STEPS)"
echo "Batch size: $BATCH_SIZE x $ACCUM_STEPS accumulation = $EFFECTIVE_BATCH effective"
echo "Config: AdamW + cosine + LR=0.01 + LSTM dim=256"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="
echo ""
echo "KEY CHANGES:"
echo "  1. Normalization stats computed per fold from training samples only"
echo "  2. Early stopping: patience=30, min_epochs=50"
echo ""

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

# Output directory (v2 suffix indicates per-fold normalization)
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/${EXPERIMENT}_v2_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling $TEMPORAL_SAMPLING \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 2 \
    --skip-aggregation max \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUM_STEPS \
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
    --early-stopping \
    --patience 30 \
    --min-epochs 50 \
    --wandb \
    --wandb-project landtake-multitemporal

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
