#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp2_input_representation/outputs/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp2_input_representation/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Validation-set inference for Part II experiments (CV-macro recomputation).
# ============================================================================
#
# Usage (all folds for one experiment):
#   sbatch --job-name=p2_val_A3 scripts/slurm/evaluate_val.sh A3_s2_9band
#
# Usage (single fold smoke test):
#   sbatch --job-name=p2_val_A3_f0 scripts/slurm/evaluate_val.sh A3_s2_9band 0
#
# ============================================================================

EXPERIMENT=${1:?"Usage: sbatch evaluate_val.sh EXPERIMENT [FOLD]"}
FOLD=${2:-}

echo "=========================================="
echo "Part II val inference: ${EXPERIMENT} fold=${FOLD:-all}"
echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

if [[ -n "$FOLD" ]]; then
    python experiments/exp2_input_representation/scripts/modeling/evaluate_val_set.py \
        --experiment "$EXPERIMENT" --fold "$FOLD"
else
    python experiments/exp2_input_representation/scripts/modeling/evaluate_val_set.py \
        --experiment "$EXPERIMENT" --all-folds
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
