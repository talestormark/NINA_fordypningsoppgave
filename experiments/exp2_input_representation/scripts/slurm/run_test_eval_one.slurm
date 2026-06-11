#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=PART2_spectral_spatial_resolution_experiments/outputs/logs/test_eval_%x_%j.log
#SBATCH --error=PART2_spectral_spatial_resolution_experiments/outputs/logs/test_eval_%x_%j.err

# Test-set inference for a single Part II experiment (probability ensemble across 5 folds).
# Usage: sbatch --job-name=p2_test_<exp> run_test_eval_one.slurm <experiment_name>

EXPERIMENT=${1:?"Usage: sbatch run_test_eval_one.slurm EXPERIMENT"}

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

echo "=== Test evaluation: ${EXPERIMENT} ==="
python PART2_spectral_spatial_resolution_experiments/scripts/modeling/evaluate_test_set.py \
    --experiment "${EXPERIMENT}" --all-folds

echo "=== Done ==="
