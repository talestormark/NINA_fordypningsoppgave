#!/bin/bash
#SBATCH --job-name=ae_rand_eval
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp3_annotation/outputs/slurm_rand_eval_%j.log

# Test-set evaluation for random-placement variants of E4 (masked-loss U-Net).
# Two configs: AE input and S-2 9-band input. RF rand variant already has test data.

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

echo "=== Random placement: AlphaEarth U-Net ==="
python experiments/exp3_annotation/scripts/modeling/evaluate_test_set.py \
    --exp E4_rand_D2_alphaearth \
    --data-experiment D2_alphaearth \
    --num-folds 5

echo ""
echo "=== Random placement: Sentinel-2 U-Net ==="
python experiments/exp3_annotation/scripts/modeling/evaluate_test_set.py \
    --exp E4_rand_A3_s2_9band \
    --data-experiment A3_s2_9band \
    --num-folds 5

echo "=== Done ==="
