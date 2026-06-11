#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=rq1_lite
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v2/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp1_temporal_sampling/outputs_v2/logs/slurm_%x_%j.err

# ============================================================================
# RQ1 Lite: Reduced-capacity LSTM on all temporal sampling conditions
# ============================================================================
#
# Motivation: v1 showed lite models (1-layer ConvLSTM, h=32, ~30.3M params)
# outperform full models (2-layer, h=256, ~54.4M) — suggesting
# over-parameterization hurts. Run lite versions of all RQ1 conditions
# on v2 data to see if reduced capacity helps.
#
# All use: 1-layer ConvLSTM, h=32, kernel=3×3 (~30.3M params)
#
# Experiments:
#   exp001_lite  Annual T=7      batch=4
#   exp002_lite  Bi-seasonal T=14  batch=2 (×2 accum)
#   exp003_lite  Bi-temporal T=2   batch=8
#
# Usage:
#   sbatch train_rq1_lite.sh <EXPERIMENT> <FOLD>
#
# Examples:
#   # Single experiment/fold
#   sbatch train_rq1_lite.sh exp001_lite 0
#
#   # All 15 jobs
#   for exp in exp001_lite exp002_lite exp003_lite; do
#     for fold in 0 1 2 3 4; do
#       sbatch train_rq1_lite.sh $exp $fold
#     done
#   done
#
# ============================================================================

EXPERIMENT=${1:-"exp001_lite"}
FOLD=${2:-"0"}

# Lite architecture: 1-layer ConvLSTM, h=32
LSTM_HIDDEN_DIM=32
LSTM_NUM_LAYERS=1
CONVLSTM_KERNEL=3

# Set experiment-specific parameters
case $EXPERIMENT in
    "exp001_lite")  # LSTM-7-lite: Annual T=7
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        ;;
    "exp002_lite")  # LSTM-14-lite: Bi-seasonal T=14
        TEMPORAL_SAMPLING="quarterly"
        TIME_STEPS=14
        BATCH_SIZE=2
        ACCUM_STEPS=2
        ;;
    "exp003_lite")  # LSTM-2-lite: Bi-temporal T=2
        TEMPORAL_SAMPLING="bi_temporal"
        TIME_STEPS=2
        BATCH_SIZE=8
        ACCUM_STEPS=1
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Valid: exp001_lite, exp002_lite, exp003_lite"
        exit 1
        ;;
esac

WANDB_PROJECT="RQ1_lite_v2"
EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUM_STEPS))

echo "=========================================="
echo "EXPERIMENT: $EXPERIMENT (RQ1-lite, data_v2)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: lstm_unet (lite: h=$LSTM_HIDDEN_DIM, layers=$LSTM_NUM_LAYERS)"
echo "Temporal sampling: $TEMPORAL_SAMPLING (T=$TIME_STEPS)"
echo "Batch size: $BATCH_SIZE x $ACCUM_STEPS accumulation = $EFFECTIVE_BATCH effective"
echo "LSTM: hidden=$LSTM_HIDDEN_DIM, layers=$LSTM_NUM_LAYERS, kernel=$CONVLSTM_KERNEL"
echo "Config: AdamW + cosine + LR=0.01 + 400 epochs (no early stopping)"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="
echo ""

# Navigate to project root
cd /cluster/home/tmstorma/NINA_fordypningsoppgave

# Create log directory
mkdir -p experiments/exp1_temporal_sampling/outputs_v2/logs

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
OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v2/${EXPERIMENT}_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run training
python experiments/exp1_temporal_sampling/scripts/modeling/train_multitemporal.py \
    --model-name lstm_unet \
    --lstm-hidden-dim $LSTM_HIDDEN_DIM \
    --lstm-num-layers $LSTM_NUM_LAYERS \
    --convlstm-kernel-size $CONVLSTM_KERNEL \
    --skip-aggregation max \
    --temporal-sampling $TEMPORAL_SAMPLING \
    --encoder-name resnet50 \
    --encoder-weights imagenet \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUM_STEPS \
    --image-size 64 \
    --num-workers 4 \
    --epochs 400 \
    --lr 0.01 \
    --optimizer adamw \
    --scheduler cosine \
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
    --data-dir data_v2 \
    --mask-subdir Land_take_masks_coarse \
    --splits-dir preprocessing/outputs/splits/part1 \
    --change-level-path preprocessing/outputs/splits/part1/split_info.csv \
    --wandb \
    --wandb-project $WANDB_PROJECT

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
