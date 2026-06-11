#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=pilot_ablation
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v2/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp1_temporal_sampling/outputs_v2/logs/slurm_%x_%j.err

# ============================================================================
# Pilot Ablations for Part 1 v2
# ============================================================================
#
# Quick experiments to isolate effects of padding, loss, and regularization.
#
# Padding + loss pilots (fold 0, 150 epochs):
#   pilot_imgsize96   — image_size=96 with reflect padding + valid mask
#   pilot_focal_only  — focal loss only + alpha=0.25 (like v1), image_size=96
#
# Regularization pilots (fold 1, 400 epochs — worst fold at 44.5% IoU):
#   pilot_early_stop  — early stopping (patience=50, min_epochs=100), image_size=96
#   pilot_wd5e3       — weight_decay=5e-3 (10x baseline), image_size=96
#   pilot_es_wd       — early stopping + weight_decay=5e-3 combined, image_size=96
#
# Baselines:
#   exp001_fold0: 60.7% best val IoU (image_size=64, focal+dice, wd=5e-4)
#   exp001_fold1: 44.5% best val IoU (same settings, hardest fold)
#
# Usage:
#   sbatch pilot_imgsize_vs_loss.sh <pilot_name>
#
#   # All at once:
#   for p in pilot_imgsize96 pilot_focal_only pilot_early_stop pilot_wd5e3 pilot_es_wd; do
#     sbatch pilot_imgsize_vs_loss.sh $p
#   done
# ============================================================================

PILOT=${1:-"pilot_imgsize96"}

cd /cluster/home/tmstorma/NINA_fordypningsoppgave
mkdir -p experiments/exp1_temporal_sampling/outputs_v2/logs

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

echo "=========================================="
echo "PILOT: $PILOT"
echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Model settings (same across all pilots = annual LSTM-7)
MODEL_ARGS="
    --model-name lstm_unet
    --temporal-sampling annual
    --encoder-name resnet50
    --encoder-weights imagenet
    --lstm-hidden-dim 256
    --lstm-num-layers 2
    --convlstm-kernel-size 3
    --skip-aggregation max
    --batch-size 4
    --accumulation-steps 1
    --num-workers 4
    --lr 0.01
    --optimizer adamw
    --scheduler cosine
    --seed 42
    --num-folds 5
    --data-dir data_v2
    --mask-subdir Land_take_masks_coarse
    --splits-dir preprocessing/outputs/splits/part1
    --change-level-path preprocessing/outputs/splits/part1/split_info.csv
    --wandb
    --wandb-project pilot_ablation
"

case $PILOT in
    # --- Padding + loss pilots (fold 0, 150 epochs) ---
    "pilot_imgsize96")
        echo "Testing: image_size=96 (full tile, with reflect padding + valid mask)"
        echo "Changed: image_size 64 -> 96"
        echo "Kept:    focal+dice, alpha=0.75, wd=5e-4, no early stopping"
        OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v2/pilot_imgsize96_fold0"
        PILOT_ARGS="--fold 0 --epochs 150 --image-size 96 --weight-decay 5e-4 --loss focal_dice --focal-alpha 0.75 --lambda-focal 1.0 --lambda-dice 1.0"
        ;;
    "pilot_focal_only")
        echo "Testing: focal-only loss with alpha=0.25 (like v1)"
        echo "Changed: loss focal_dice -> focal, alpha 0.75 -> 0.25"
        echo "Kept:    image_size=96, wd=5e-4, no early stopping"
        OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v2/pilot_focal_only_fold0"
        PILOT_ARGS="--fold 0 --epochs 150 --image-size 96 --weight-decay 5e-4 --loss focal --focal-alpha 0.25"
        ;;
    # --- Regularization pilots (fold 1, 400 epochs — worst fold) ---
    "pilot_early_stop")
        echo "Testing: early stopping (patience=50, min_epochs=100)"
        echo "Changed: --early-stopping --patience 50 --min-epochs 100"
        echo "Kept:    image_size=96, focal+dice, alpha=0.75, wd=5e-4"
        OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v2/pilot_early_stop_fold1"
        PILOT_ARGS="--fold 1 --epochs 400 --image-size 96 --weight-decay 5e-4 --loss focal_dice --focal-alpha 0.75 --lambda-focal 1.0 --lambda-dice 1.0 --early-stopping --patience 50 --min-epochs 100"
        ;;
    "pilot_wd5e3")
        echo "Testing: stronger weight decay (5e-3, 10x baseline)"
        echo "Changed: weight_decay 5e-4 -> 5e-3"
        echo "Kept:    image_size=96, focal+dice, alpha=0.75, no early stopping"
        OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v2/pilot_wd5e3_fold1"
        PILOT_ARGS="--fold 1 --epochs 400 --image-size 96 --weight-decay 5e-3 --loss focal_dice --focal-alpha 0.75 --lambda-focal 1.0 --lambda-dice 1.0"
        ;;
    "pilot_es_wd")
        echo "Testing: early stopping + stronger weight decay combined"
        echo "Changed: --early-stopping --patience 50 --min-epochs 100, wd=5e-3"
        echo "Kept:    image_size=96, focal+dice, alpha=0.75"
        OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v2/pilot_es_wd_fold1"
        PILOT_ARGS="--fold 1 --epochs 400 --image-size 96 --weight-decay 5e-3 --loss focal_dice --focal-alpha 0.75 --lambda-focal 1.0 --lambda-dice 1.0 --early-stopping --patience 50 --min-epochs 100"
        ;;
    *)
        echo "Unknown pilot: $PILOT"
        echo "Valid: pilot_imgsize96, pilot_focal_only, pilot_early_stop, pilot_wd5e3, pilot_es_wd"
        exit 1
        ;;
esac

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

python experiments/exp1_temporal_sampling/scripts/modeling/train_multitemporal.py \
    $MODEL_ARGS \
    $PILOT_ARGS \
    --output-dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
