#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=p2_train
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
#   sbatch train_experiment.sh EXPERIMENT FOLD [EXTRA_ARGS...]
#
# Examples:
#   sbatch train_experiment.sh A3_s2_9band 0
#   sbatch train_experiment.sh D3_s2_ae_fusion 2
#   sbatch train_experiment.sh A3_s2_9band 0 --model-name lstm_unet --lstm-hidden-dim 256 --lstm-num-layers 2
#
# Submit all 5 folds:
#   for fold in 0 1 2 3 4; do sbatch train_experiment.sh A3_s2_9band $fold; done
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
    --model-name late_fusion_pool \
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
    --warmup-epochs 5 \
    --weight-decay 5e-4 \
    --loss focal \
    --focal-alpha 0.25 \
    --focal-gamma 2.0 \
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
