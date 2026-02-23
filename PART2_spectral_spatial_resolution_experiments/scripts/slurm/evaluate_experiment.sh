#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=p2_eval
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART2_spectral_spatial_resolution_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART2_spectral_spatial_resolution_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Batch test set evaluation for Part II experiments
# ============================================================================
#
# Usage:
#   # Single experiment, all folds + ensemble + save predictions
#   sbatch evaluate_experiment.sh A3_s2_9band
#
#   # All completed experiments
#   for exp in A3_s2_9band LSTM7lite_sanity; do
#       sbatch evaluate_experiment.sh $exp
#   done
#
# ============================================================================

EXPERIMENT=${1:?"Usage: sbatch evaluate_experiment.sh EXPERIMENT"}

echo "=========================================="
echo "Part II Evaluation: ${EXPERIMENT}"
echo "=========================================="
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

# Run evaluation with all folds + ensemble + save predictions
python PART2_spectral_spatial_resolution_experiments/scripts/modeling/evaluate_test_set.py \
    --experiment $EXPERIMENT \
    --all-folds \
    --num-folds 5 \
    --batch-size 1 \
    --save-predictions

# Run boundary metrics on saved predictions
echo ""
echo "=========================================="
echo "Computing boundary metrics..."
echo "=========================================="

for fold in 0 1 2 3 4; do
    PRED_DIR="PART2_spectral_spatial_resolution_experiments/outputs/experiments/${EXPERIMENT}_fold${fold}/predictions"
    if [ -d "$PRED_DIR" ]; then
        echo "  Fold ${fold}: ${PRED_DIR}"
        python PART2_spectral_spatial_resolution_experiments/scripts/analysis/boundary_metrics.py \
            --predictions-dir $PRED_DIR \
            --tolerances 1 2
    fi
done

# Ensemble predictions
ENS_DIR="PART2_spectral_spatial_resolution_experiments/outputs/experiments/${EXPERIMENT}/ensemble_predictions"
if [ -d "$ENS_DIR" ]; then
    echo "  Ensemble: ${ENS_DIR}"
    python PART2_spectral_spatial_resolution_experiments/scripts/analysis/boundary_metrics.py \
        --predictions-dir $ENS_DIR \
        --tolerances 1 2
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
