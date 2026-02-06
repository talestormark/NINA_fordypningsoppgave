#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp004_v2
#SBATCH --time=0-02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# LSTM-1x1: Kernel ablation (T=7) — unified 400-epoch protocol
# ============================================================================
#
# Re-train with the same protocol as all architecture baselines:
#   400 epochs, NO early stopping, cosine annealing, AdamW
#
# Usage:
#   for fold in 0 1 2 3 4; do sbatch retrain_exp004_1x1.sh $fold; done
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "LSTM-1x1: Kernel ablation (T=7) — 400 epochs, no ES"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "ConvLSTM kernel_size: 1 (per-pixel temporal modeling)"
echo "Batch size: 4"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/exp004_v2_fold${FOLD}"
echo "Output directory: $OUTPUT_DIR"
echo ""

python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling annual \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 256 \
    --lstm-num-layers 2 \
    --convlstm-kernel-size 1 \
    --skip-aggregation max \
    --batch-size 4 \
    --accumulation-steps 1 \
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
