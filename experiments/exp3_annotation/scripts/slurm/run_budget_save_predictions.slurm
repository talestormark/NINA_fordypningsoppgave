#!/bin/bash
#SBATCH --job-name=ae_budget_preds
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp3_annotation/outputs/slurm_budget_preds_%j.log

# Regenerate per-tile ensemble prediction maps for the point-budget variants of
# E4 (masked-loss U-Net on AlphaEarth), so the qualitative saturation figure can
# render them. Runs 5-fold inference + probability-averaged ensemble for
# n5 (10 pts), n10 (20 pts), n15 (30 pts), n100 (200 pts) and writes
# ensemble_predictions/{refid}.npz (prob float16, mask uint8) under each config.
# (50 pts = E4_ae_unet_sparse and dense = D2_alphaearth already have predictions.)

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

echo "=== Budget variant prediction-map regeneration ==="
python experiments/exp3_annotation/scripts/modeling/evaluate_test_set.py \
    --exp E4_D2_alphaearth_sparse_n5 \
          E4_D2_alphaearth_sparse_n10 \
          E4_D2_alphaearth_sparse_n15 \
          E4_D2_alphaearth_sparse_n100 \
    --data-experiment D2_alphaearth \
    --num-folds 5 \
    --save-predictions

echo "=== Done ==="
