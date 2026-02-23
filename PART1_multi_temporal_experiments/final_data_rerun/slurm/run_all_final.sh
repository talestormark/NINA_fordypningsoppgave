#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=final_%1_%2
#SBATCH --time=0-03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/final_data_rerun/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/final_data_rerun/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Part 1 Final Re-run — EPSG:3035 Native Data
# ============================================================================
#
# Consolidated launcher for all 10 experiments on clean EPSG:3035 data.
# All experiments use 5-epoch LR warmup + cosine annealing.
#
# Usage:
#   sbatch run_all_final.sh <EXPERIMENT> <FOLD>
#
# Examples:
#   sbatch run_all_final.sh exp010 0          # Anchor, fold 0
#   for f in 0 1 2 3 4; do sbatch run_all_final.sh exp010 $f; done
#
# ============================================================================

EXP=${1:?"Usage: sbatch run_all_final.sh <EXPERIMENT> <FOLD>"}
FOLD=${2:?"Usage: sbatch run_all_final.sh <EXPERIMENT> <FOLD>"}

# ── Shared configuration ────────────────────────────────────────────────────
REPO=/cluster/home/tmstorma/NINA_fordypningsoppgave
DATA_DIR=${REPO}/data/processed/epsg3035_10m_v1
BASE_OUTPUT=${REPO}/PART1_multi_temporal_experiments/final_data_rerun/outputs/experiments
TRAIN_SCRIPT=${REPO}/PART1_multi_temporal_experiments/scripts/modeling/train_multitemporal.py
WANDB_PROJECT=landtake-multitemporal-final
WARMUP=5
EPOCHS=400
LR=0.01
OPTIMIZER=adamw
SCHEDULER=cosine
LOSS=focal
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0
IMAGE_SIZE=64
SEED=42
NUM_FOLDS=5
ENCODER=resnet50
ENCODER_WEIGHTS=imagenet
SKIP_AGG=max
WEIGHT_DECAY=5e-4

# ── Per-experiment configuration ─────────────────────────────────────────────
case $EXP in
  exp010)
    # LSTM-7 (anchor) — 2-layer ConvLSTM h=256 k=3, annual T=7
    EXP_NAME="exp010_lstm7_no_es"
    MODEL=lstm_unet
    SAMPLING=annual
    BATCH_SIZE=4
    ACCUM=1
    LSTM_H=256
    LSTM_L=2
    LSTM_K=3
    ;;
  exp002)
    # LSTM-14 — quarterly T=14
    EXP_NAME="exp002_v3"
    MODEL=lstm_unet
    SAMPLING=quarterly
    BATCH_SIZE=2
    ACCUM=2
    LSTM_H=256
    LSTM_L=2
    LSTM_K=3
    ;;
  exp003)
    # LSTM-2 — bi-temporal T=2
    EXP_NAME="exp003_v3"
    MODEL=lstm_unet
    SAMPLING=bi_temporal
    BATCH_SIZE=8
    ACCUM=1
    LSTM_H=256
    LSTM_L=2
    LSTM_K=3
    ;;
  exp004)
    # LSTM-1x1 — 2-layer ConvLSTM h=256 k=1, annual T=7
    EXP_NAME="exp004_v2"
    MODEL=lstm_unet
    SAMPLING=annual
    BATCH_SIZE=4
    ACCUM=1
    LSTM_H=256
    LSTM_L=2
    LSTM_K=1
    ;;
  exp005)
    # EarlyFusion — stacked-channel U-Net, bi-temporal T=2
    EXP_NAME="exp005_early_fusion"
    MODEL=early_fusion_unet
    SAMPLING=bi_temporal
    BATCH_SIZE=8
    ACCUM=1
    LSTM_H=256   # unused but set for CLI
    LSTM_L=2
    LSTM_K=3
    ;;
  exp006)
    # LateFusion — shared encoder + concat, bi-temporal T=2
    EXP_NAME="exp006_late_fusion"
    MODEL=late_fusion_concat
    SAMPLING=bi_temporal
    BATCH_SIZE=8
    ACCUM=1
    LSTM_H=256
    LSTM_L=2
    LSTM_K=3
    ;;
  exp007)
    # Pool-7 — shared encoder + mean pool, annual T=7
    EXP_NAME="exp007_late_fusion_pool"
    MODEL=late_fusion_pool
    SAMPLING=annual
    BATCH_SIZE=4
    ACCUM=1
    LSTM_H=256
    LSTM_L=2
    LSTM_K=3
    ;;
  exp008)
    # Conv3D-7 — shared encoder + 3D conv, annual T=7
    EXP_NAME="exp008_conv3d_fusion"
    MODEL=conv3d_fusion
    SAMPLING=annual
    BATCH_SIZE=4
    ACCUM=1
    LSTM_H=256
    LSTM_L=2
    LSTM_K=3
    ;;
  exp009)
    # LSTM-2-lite — 1-layer ConvLSTM h=32 k=3, bi-temporal T=2
    EXP_NAME="exp009_lstm_lite"
    MODEL=lstm_unet
    SAMPLING=bi_temporal
    BATCH_SIZE=8
    ACCUM=1
    LSTM_H=32
    LSTM_L=1
    LSTM_K=3
    ;;
  exp011)
    # LSTM-7-lite — 1-layer ConvLSTM h=32 k=3, annual T=7
    EXP_NAME="exp011_lstm7_lite"
    MODEL=lstm_unet
    SAMPLING=annual
    BATCH_SIZE=4
    ACCUM=1
    LSTM_H=32
    LSTM_L=1
    LSTM_K=3
    ;;
  *)
    echo "ERROR: Unknown experiment '$EXP'"
    echo "Valid experiments: exp002 exp003 exp004 exp005 exp006 exp007 exp008 exp009 exp010 exp011"
    exit 1
    ;;
esac

OUTPUT_DIR="${BASE_OUTPUT}/${EXP_NAME}_fold${FOLD}"

echo "=========================================="
echo "FINAL RE-RUN: ${EXP} (${EXP_NAME})"
echo "=========================================="
echo "Fold: ${FOLD} / $((NUM_FOLDS - 1))"
echo "Model: ${MODEL}"
echo "Sampling: ${SAMPLING}"
echo "Batch size: ${BATCH_SIZE} (accum: ${ACCUM}, effective: $((BATCH_SIZE * ACCUM)))"
echo "LSTM: h=${LSTM_H}, layers=${LSTM_L}, k=${LSTM_K}"
echo "Warmup: ${WARMUP} epochs"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Navigate to project root
cd ${REPO}

# Load environment
module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training
python ${TRAIN_SCRIPT} \
    --model-name ${MODEL} \
    --temporal-sampling ${SAMPLING} \
    --encoder-name ${ENCODER} \
    --encoder-weights ${ENCODER_WEIGHTS} \
    --lstm-hidden-dim ${LSTM_H} \
    --lstm-num-layers ${LSTM_L} \
    --convlstm-kernel-size ${LSTM_K} \
    --skip-aggregation ${SKIP_AGG} \
    --batch-size ${BATCH_SIZE} \
    --accumulation-steps ${ACCUM} \
    --image-size ${IMAGE_SIZE} \
    --num-workers 4 \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --optimizer ${OPTIMIZER} \
    --scheduler ${SCHEDULER} \
    --warmup-epochs ${WARMUP} \
    --weight-decay ${WEIGHT_DECAY} \
    --loss ${LOSS} \
    --focal-alpha ${FOCAL_ALPHA} \
    --focal-gamma ${FOCAL_GAMMA} \
    --output-dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --fold ${FOLD} \
    --num-folds ${NUM_FOLDS} \
    --data-dir ${DATA_DIR} \
    --wandb \
    --wandb-project ${WANDB_PROJECT}

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
