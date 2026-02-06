#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=exp004_1x1
#SBATCH --time=0-02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Experiment 004: 1D vs 2D Temporal Modeling (RQ2c)
# ============================================================================
#
# This experiment compares per-pixel (1x1) vs patch-based (3x3) ConvLSTM
# kernel sizes to investigate how spatial context in temporal modeling
# affects segmentation accuracy and robustness.
#
# Ablation: ConvLSTM kernel_size = 1 (per-pixel temporal modeling)
# Baseline: exp001_v2 uses kernel_size = 3 (patch-based, already complete)
#
# All other settings identical to exp001_v2 (annual sampling, T=7)
#
# Usage:
#   sbatch train_exp004_1x1.sh <FOLD>
#   FOLD: 0, 1, 2, 3, or 4 (for 5-fold CV)
#
# Examples:
#   # Run single fold
#   sbatch train_exp004_1x1.sh 0
#
#   # Run all folds
#   for fold in 0 1 2 3 4; do sbatch train_exp004_1x1.sh $fold; done
#
# ============================================================================

FOLD=${1:-"0"}

# Fixed settings for this experiment (same as exp001 annual baseline)
TEMPORAL_SAMPLING="annual"
TIME_STEPS=7
BATCH_SIZE=4
ACCUM_STEPS=1
CONVLSTM_KERNEL_SIZE=1  # Per-pixel temporal modeling

EFFECTIVE_BATCH=$((BATCH_SIZE * ACCUM_STEPS))

echo "=========================================="
echo "EXPERIMENT 004: 1D Temporal Modeling (k=1x1)"
echo "=========================================="
echo "Research Question: RQ2c"
echo "Ablation: ConvLSTM kernel_size = $CONVLSTM_KERNEL_SIZE (per-pixel)"
echo "Fold: $FOLD / 4"
echo "Temporal sampling: $TEMPORAL_SAMPLING (T=$TIME_STEPS)"
echo "Batch size: $BATCH_SIZE x $ACCUM_STEPS accumulation = $EFFECTIVE_BATCH effective"
echo "Config: AdamW + cosine + LR=0.01 + LSTM dim=256"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="
echo ""
echo "COMPARISON:"
echo "  Baseline (exp001_v2): kernel_size=3 (3x3 patch-based)"
echo "  This experiment:      kernel_size=1 (1x1 per-pixel)"
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

# Output directory
OUTPUT_DIR="multi_temporal_experiments/outputs/experiments/exp004_1x1_fold${FOLD}"

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
    --convlstm-kernel-size $CONVLSTM_KERNEL_SIZE \
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
