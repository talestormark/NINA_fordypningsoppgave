#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp009_lite
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 009: ConvLSTM-lite (T=2, ~31M params)
# ============================================================================
#
# Research Question: Is boundary degradation in exp003 from recurrence or
#   over-parameterisation? Parameter-matched ConvLSTM isolates the effect.
#
# Configuration:
#   - Model: LSTM-UNet with reduced ConvLSTM (1 layer, hidden_dim=32)
#   - Encoder: ResNet-50 (ImageNet pretrained, shared across timesteps)
#   - Temporal sampling: bi_temporal (T=2, 2018 Q2, 2024 Q3)
#   - ConvLSTM: 1 layer, hidden_dim=32 (vs 2 layers, 256 in exp003)
#   - Parameters: ~30.4M (vs ~54.4M in exp003, matches baselines)
#   - 5-fold stratified cross-validation
#
# Comparisons:
#   exp009 vs exp003: Lite LSTM vs full LSTM at T=2
#     If lite wins on BF: over-parameterisation caused boundary degradation
#     If full wins: full LSTM needed despite param cost
#
#   exp009 vs exp005/006: Lite LSTM vs non-temporal baselines at T=2
#     If lite wins: recurrence helps even parameter-matched
#     If baselines win/tie: recurrence still not helpful
#
# Usage:
#   sbatch train_exp009_lstm_lite.sh [FOLD]
#
# Example:
#   for fold in 0 1 2 3 4; do sbatch train_exp009_lstm_lite.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

echo "=========================================="
echo "EXPERIMENT 009: ConvLSTM-lite (T=2)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: LSTM-UNet with 1-layer ConvLSTM, hidden_dim=32"
echo "Temporal sampling: bi_temporal (T=2, 2018 Q2, 2024 Q3)"
echo "Config: AdamW + cosine + LR=0.01"
echo "Batch size: 8"
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
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/exp009_lstm_lite_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python multi_temporal_experiments/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --temporal-sampling bi_temporal \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --lstm-hidden-dim 32 \
    --lstm-num-layers 1 \
    --convlstm-kernel-size 3 \
    --skip-aggregation max \
    --batch-size 8 \
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
