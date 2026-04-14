#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART2_spectral_spatial_resolution_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART2_spectral_spatial_resolution_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Generic training script for Part II experiments
# ============================================================================
#
# Usage:
#   sbatch --job-name=p2_A3_f0 train_experiment.sh A3_s2_9band 0
#
# Submit all 5 folds:
#   for fold in 0 1 2 3 4; do
#     sbatch --job-name=p2_A3_f${fold} train_experiment.sh A3_s2_9band $fold
#   done
#
# Helper to submit all folds with auto-naming:
#   EXP=A1_s2_rgb; for f in 0 1 2 3 4; do sbatch --job-name=p2_${EXP}_f${f} train_experiment.sh $EXP $f; done
#
# ============================================================================

EXPERIMENT=${1:?"Usage: sbatch train_experiment.sh EXPERIMENT FOLD [EXTRA_ARGS...]"}
FOLD=${2:?"Usage: sbatch train_experiment.sh EXPERIMENT FOLD [EXTRA_ARGS...]"}
shift 2
EXTRA_ARGS="$@"

echo "=========================================="
echo "Part II: ${EXPERIMENT} fold ${FOLD}"
echo "=========================================="
echo "Extra args: ${EXTRA_ARGS}"
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
OUTPUT_DIR="PART2_spectral_spatial_resolution_experiments/outputs/experiments/${EXPERIMENT}_fold${FOLD}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python PART2_spectral_spatial_resolution_experiments/scripts/modeling/train.py \
    --experiment $EXPERIMENT \
    --model-name early_fusion_unet \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --batch-size 8 \
    --image-size 64 \
    --num-workers 4 \
    --epochs 400 \
    --lr 0.01 \
    --optimizer adamw \
    --scheduler cosine \
    --warmup-epochs 0 \
    --weight-decay 5e-4 \
    --loss focal_dice \
    --focal-alpha 0.75 \
    --focal-gamma 2.0 \
    --lambda-focal 1.0 \
    --lambda-dice 1.0 \
    --output-dir $OUTPUT_DIR \
    --seed 42 \
    --fold $FOLD \
    --num-folds 5 \
    --wandb \
    --wandb-project landtake-spectral-spatial \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
