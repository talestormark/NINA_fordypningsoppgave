#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp1_temporal_sampling/outputs_v3/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp1_temporal_sampling/outputs_v3/logs/slurm_%x_%j.err

# ============================================================================
# Per-fold test-set inference for Part I experiments.
# Saves per-fold per-tile IoU so the Test column in results tables can show
# ± std (across the five single-fold models, no ensembling).
# ============================================================================

EXPERIMENT=${1:?"Usage: sbatch evaluate_test_per_fold.sh EXPERIMENT [FOLD]"}
FOLD=${2:-}

echo "Part I per-fold test inference: ${EXPERIMENT} fold=${FOLD:-all}"
echo "Job started at: $(date)"

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

if [[ -n "$FOLD" ]]; then
    python experiments/exp1_temporal_sampling/scripts/modeling/evaluate_test_per_fold.py \
        --experiment "$EXPERIMENT" --fold "$FOLD"
else
    python experiments/exp1_temporal_sampling/scripts/modeling/evaluate_test_per_fold.py \
        --experiment "$EXPERIMENT" --all-folds
fi

echo "Job finished at: $(date)"
