#!/bin/bash
#SBATCH --job-name=ae_budget_eval
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp3_annotation/outputs/slurm_budget_eval_%j.log

# Test-set evaluation for the point-budget variants of E4 (masked-loss U-Net on AlphaEarth).
# Runs 5-fold inference + ensemble for n5, n10, n100 and writes test_results.json
# under each experiment's output directory.

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

echo "=== Budget variant test evaluation ==="
python experiments/exp3_annotation/scripts/modeling/evaluate_test_set.py \
    --exp E4_D2_alphaearth_sparse_n5 \
          E4_D2_alphaearth_sparse_n10 \
          E4_D2_alphaearth_sparse_n100 \
    --data-experiment D2_alphaearth \
    --num-folds 5

echo "=== Done ==="
