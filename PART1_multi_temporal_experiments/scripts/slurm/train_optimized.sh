#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=lstm_opt
#SBATCH --time=0-01:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# Phase 2: Full 5-fold CV with optimized hyperparameters
# Best config from Phase 1: AdamW + cosine + LR=0.01 + LSTM dim 256 → 61.66% IoU

FOLD=${1:-"0"}

echo "=========================================="
echo "PHASE 2: Optimized LSTM-UNet Training"
echo "Fold: $FOLD / 4"
echo "Config: AdamW + cosine + LR=0.01 + LSTM dim 256"
echo "Job started at: $(date)"
echo "=========================================="

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

OUTPUT_DIR="PART1_multi_temporal_experiments/outputs/experiments/exp001_optimized_fold${FOLD}"

python PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 2 \
    --skip-aggregation max \
    --batch-size 4 \
    --image-size 64 \
    --num-workers 4 \
    --epochs 200 \
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

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
