#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v3/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp1_temporal_sampling/outputs_v3/logs/slurm_%x_%j.err
# NOTE: Pass --job-name on sbatch command line for descriptive names:
#   sbatch --job-name=p1_exp001_f0 train_all_experiments_v3.sh exp001 0

# ============================================================================
# Re-run All Part 1 Experiments on data_v2
# ============================================================================
#
# Full re-run of all 10 v1 experiments on the larger dataset.
#
# Data:
#   - data_v2: 163 full-window tiles (EPSG:3035, square 10m pixels)
#   - Splits: preprocessing/outputs/splits/part1/
#   - Masks: Land_take_masks_coarse/
#   - 400 epochs, no early stopping, 6h time limit
#
# Usage:
#   sbatch train_all_experiments_v3.sh <EXPERIMENT> <FOLD>
#
# Experiments:
#   RQ1 (temporal sampling):
#     exp001  LSTM-7       Annual T=7    (2-layer ConvLSTM h=256, 3x3)
#     exp002  LSTM-14      Bi-seasonal T=14
#     exp003  LSTM-2       Bi-temporal T=2
#
#   RQ2a-b (bi-temporal baselines):
#     exp005  EarlyFusion  Bi-temporal T=2  (stacked channels, no shared encoder)
#     exp006  LateFusion   Bi-temporal T=2  (shared encoder + concat)
#
#   RQ2c-d (T=7 architecture comparisons):
#     exp007  Pool-7       Annual T=7  (mean pool fusion)
#     exp008  Conv3D-7     Annual T=7  (3D conv fusion)
#
#   RQ2e (parameter-matched controls):
#     exp009  LSTM-2-lite  Bi-temporal T=2  (1-layer ConvLSTM h=32)
#     exp010  LSTM-7-lite  Annual T=7       (1-layer ConvLSTM h=32)
#
#   RQ2f (kernel ablation):
#     exp004  LSTM-1x1     Annual T=7  (2-layer ConvLSTM h=256, 1x1)
#
# Examples:
#   # Single experiment/fold
#   sbatch train_all_experiments_v3.sh exp001 0
#
#   # All folds for one experiment
#   for fold in 0 1 2 3 4; do sbatch train_all_experiments_v3.sh exp001 $fold; done
#
#   # Phase 1: RQ1 (15 jobs)
#   for exp in exp001 exp002 exp003; do
#     for fold in 0 1 2 3 4; do
#       sbatch train_all_experiments_v3.sh $exp $fold
#     done
#   done
#
#   # Phase 2: RQ2 (35 jobs)
#   for exp in exp004 exp005 exp006 exp007 exp008 exp009 exp010; do
#     for fold in 0 1 2 3 4; do
#       sbatch train_all_experiments_v3.sh $exp $fold
#     done
#   done
#
#   # All experiments (50 jobs)
#   for exp in exp001 exp002 exp003 exp004 exp005 exp006 exp007 exp008 exp009 exp010; do
#     for fold in 0 1 2 3 4; do
#       sbatch train_all_experiments_v3.sh $exp $fold
#     done
#   done
#
# ============================================================================

EXPERIMENT=${1:-"exp001"}
FOLD=${2:-"0"}

# Defaults (overridden per experiment as needed)
LSTM_HIDDEN_DIM=256
LSTM_NUM_LAYERS=2
CONVLSTM_KERNEL=3
EXTRA_ARGS=""

# Set experiment-specific parameters
case $EXPERIMENT in
    "exp001")  # LSTM-7: Annual T=7
        MODEL_NAME="lstm_unet"
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        ;;
    "exp002")  # LSTM-14: Bi-seasonal T=14
        MODEL_NAME="lstm_unet"
        TEMPORAL_SAMPLING="quarterly"
        TIME_STEPS=14
        BATCH_SIZE=2
        ACCUM_STEPS=2
        ;;
    "exp003")  # LSTM-2: Bi-temporal T=2
        MODEL_NAME="lstm_unet"
        TEMPORAL_SAMPLING="bi_temporal"
        TIME_STEPS=2
        BATCH_SIZE=8
        ACCUM_STEPS=1
        ;;
    "exp004")  # LSTM-1x1: Annual T=7, 1x1 kernel
        MODEL_NAME="lstm_unet"
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        CONVLSTM_KERNEL=1
        ;;
    "exp005")  # EarlyFusion: Bi-temporal T=2
        MODEL_NAME="early_fusion_unet"
        TEMPORAL_SAMPLING="bi_temporal"
        TIME_STEPS=2
        BATCH_SIZE=8
        ACCUM_STEPS=1
        ;;
    "exp006")  # LateFusion: Bi-temporal T=2
        MODEL_NAME="late_fusion_concat"
        TEMPORAL_SAMPLING="bi_temporal"
        TIME_STEPS=2
        BATCH_SIZE=8
        ACCUM_STEPS=1
        ;;
    "exp007")  # Pool-7: Annual T=7
        MODEL_NAME="late_fusion_pool"
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        ;;
    "exp008")  # Conv3D-7: Annual T=7
        MODEL_NAME="conv3d_fusion"
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        ;;
    "exp009")  # LSTM-2-lite: Bi-temporal T=2, reduced capacity
        MODEL_NAME="lstm_unet"
        TEMPORAL_SAMPLING="bi_temporal"
        TIME_STEPS=2
        BATCH_SIZE=8
        ACCUM_STEPS=1
        LSTM_HIDDEN_DIM=32
        LSTM_NUM_LAYERS=1
        ;;
    "exp010")  # LSTM-7-lite: Annual T=7, reduced capacity
        MODEL_NAME="lstm_unet"
        TEMPORAL_SAMPLING="annual"
        TIME_STEPS=7
        BATCH_SIZE=4
        ACCUM_STEPS=1
        LSTM_HIDDEN_DIM=32
        LSTM_NUM_LAYERS=1
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Valid: exp001-exp010"
        exit 1
        ;;
esac

# Set WandB project based on phase
case $EXPERIMENT in
    "exp001"|"exp002"|"exp003")
        WANDB_PROJECT="RQ1_v2"
        ;;
    *)
        WANDB_PROJECT="RQ2_v2"
        ;;
esac

EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUM_STEPS))

echo "=========================================="
echo "EXPERIMENT: $EXPERIMENT (data_v2, EPSG:3035)"
echo "=========================================="
echo "Fold: $FOLD / 4"
echo "Model: $MODEL_NAME"
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
OUTPUT_DIR="experiments/exp1_temporal_sampling/outputs_v3/${EXPERIMENT}_fold${FOLD}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Build model-specific args
MODEL_ARGS="--model-name $MODEL_NAME"
if [ "$MODEL_NAME" = "lstm_unet" ]; then
    MODEL_ARGS="$MODEL_ARGS --lstm-hidden-dim $LSTM_HIDDEN_DIM --lstm-num-layers $LSTM_NUM_LAYERS --convlstm-kernel-size $CONVLSTM_KERNEL --skip-aggregation max"
fi

# Run training
python experiments/exp1_temporal_sampling/scripts/modeling/train_multitemporal.py \
    $MODEL_ARGS \
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
    --splits-dir preprocessing/outputs/splits/unified_fullwindow \
    --change-level-path preprocessing/outputs/splits/unified_fullwindow/split_info.csv \
    --wandb \
    --wandb-project $WANDB_PROJECT

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
