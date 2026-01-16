#!/bin/bash
# Sweep script to launch temporal sampling experiments with k-fold cross-validation
#
# Usage:
#   bash sweep_temporal_sampling.sh [model] [mode] [value]
#
# Modes:
#   kfold    - Launch k-fold CV for specified samplings (default)
#   seed     - Launch single-split experiments with multiple seeds (legacy)
#
# Examples:
#   # Launch 5-fold CV for all sampling modes (15 jobs total)
#   bash sweep_temporal_sampling.sh lstm_unet kfold all
#
#   # Launch 5-fold CV for annual sampling only (5 jobs)
#   bash sweep_temporal_sampling.sh lstm_unet kfold annual
#
#   # Launch seed-based experiments (legacy, no CV)
#   bash sweep_temporal_sampling.sh lstm_unet seed "42 123 456"

MODEL=${1:-"lstm_unet"}
MODE=${2:-"kfold"}
VALUE=${3:-"all"}

# Temporal sampling modes to test
if [ "$VALUE" = "all" ]; then
    SAMPLINGS=("bi_temporal" "annual" "quarterly")
elif [[ "$VALUE" =~ ^(bi_temporal|annual|quarterly)$ ]]; then
    SAMPLINGS=("$VALUE")
else
    # For seed mode, VALUE is the seed list
    SAMPLINGS=("bi_temporal" "annual" "quarterly")
fi

# Seed for k-fold split generation (always use 42 for reproducibility)
SEED=42

# Number of folds for k-fold CV
NUM_FOLDS=5

echo "=========================================="
echo "TEMPORAL SAMPLING SWEEP"
echo "=========================================="
echo "Model: $MODEL"
echo "Mode: $MODE"

if [ "$MODE" = "kfold" ]; then
    echo "Samplings: ${SAMPLINGS[@]}"
    echo "Folds: 0-$(($NUM_FOLDS - 1)) (${NUM_FOLDS}-fold CV)"
    echo "Seed: $SEED (for fold generation)"
    echo ""
    echo "This will launch $(( ${#SAMPLINGS[@]} * $NUM_FOLDS )) jobs"
elif [ "$MODE" = "seed" ]; then
    SEEDS=${VALUE:-"42 123 456"}
    echo "Samplings: ${SAMPLINGS[@]}"
    echo "Seeds: $SEEDS"
    echo ""
    echo "This will launch $(( ${#SAMPLINGS[@]} * $(echo $SEEDS | wc -w) )) jobs"
else
    echo "ERROR: Unknown mode '$MODE'. Use 'kfold' or 'seed'."
    exit 1
fi

echo ""

# Ask for confirmation
read -p "Proceed with job submission? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Launch jobs
JOB_IDS=()
SUBMITTED=0

if [ "$MODE" = "kfold" ]; then
    # K-Fold Cross-Validation Mode
    for SAMPLING in "${SAMPLINGS[@]}"; do
        # Adjust batch size for memory (quarterly needs smaller batches)
        if [ "$SAMPLING" = "quarterly" ]; then
            BATCH_SIZE=2
        elif [ "$SAMPLING" = "annual" ]; then
            BATCH_SIZE=4
        else  # bi_temporal
            BATCH_SIZE=8
        fi

        for FOLD in $(seq 0 $(($NUM_FOLDS - 1))); do
            echo ""
            echo "Submitting: $MODEL, $SAMPLING, fold $FOLD/$((NUM_FOLDS-1))"
            echo "  Batch size: $BATCH_SIZE"

            # Submit job with fold parameter
            JOB_ID=$(sbatch --parsable \
                multi_temporal_experiments/scripts/slurm/train_lstm_unet.sh \
                $MODEL $SAMPLING resnet50 $SEED $BATCH_SIZE 200 512 2 true 64 0.001 $FOLD $NUM_FOLDS)

            if [ $? -eq 0 ]; then
                JOB_IDS+=($JOB_ID)
                SUBMITTED=$((SUBMITTED + 1))
                echo "  → Job ID: $JOB_ID [SUBMITTED]"
            else
                echo "  → ERROR: Job submission failed"
            fi

            # Small delay to avoid overwhelming scheduler
            sleep 2
        done
    done

elif [ "$MODE" = "seed" ]; then
    # Seed-based Mode (Legacy, no k-fold CV)
    SEEDS=$VALUE
    for SAMPLING in "${SAMPLINGS[@]}"; do
        # Adjust batch size for memory
        if [ "$SAMPLING" = "quarterly" ]; then
            BATCH_SIZE=2
        elif [ "$SAMPLING" = "annual" ]; then
            BATCH_SIZE=4
        else  # bi_temporal
            BATCH_SIZE=8
        fi

        for SEED in $SEEDS; do
            echo ""
            echo "Submitting: $MODEL, $SAMPLING, seed $SEED"
            echo "  Batch size: $BATCH_SIZE"

            # Submit job without fold parameter (uses original split)
            JOB_ID=$(sbatch --parsable \
                multi_temporal_experiments/scripts/slurm/train_lstm_unet.sh \
                $MODEL $SAMPLING resnet50 $SEED $BATCH_SIZE 200 512 2 true 64 0.001 none 5)

            if [ $? -eq 0 ]; then
                JOB_IDS+=($JOB_ID)
                SUBMITTED=$((SUBMITTED + 1))
                echo "  → Job ID: $JOB_ID [SUBMITTED]"
            else
                echo "  → ERROR: Job submission failed"
            fi

            # Small delay to avoid overwhelming scheduler
            sleep 2
        done
    done
fi

echo ""
echo "=========================================="
echo "SUBMISSION COMPLETE"
echo "=========================================="
echo "Submitted: $SUBMITTED jobs"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f multi_temporal_experiments/outputs/logs/slurm_*.log"
echo ""
echo "Expected output directories:"
if [ "$MODE" = "kfold" ]; then
    for SAMPLING in "${SAMPLINGS[@]}"; do
        for FOLD in $(seq 0 $(($NUM_FOLDS - 1))); do
            echo "  multi_temporal_experiments/outputs/experiments/exp001_${MODEL}_${SAMPLING}_seed${SEED}_fold${FOLD}/"
        done
    done
else
    for SAMPLING in "${SAMPLINGS[@]}"; do
        for SEED in $SEEDS; do
            echo "  multi_temporal_experiments/outputs/experiments/exp001_${MODEL}_${SAMPLING}_seed${SEED}/"
        done
    done
fi
echo "=========================================="
