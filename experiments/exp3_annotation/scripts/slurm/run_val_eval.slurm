#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp3_annotation/outputs/slurm_val_eval_%x_%j.log
#SBATCH --error=experiments/exp3_annotation/outputs/slurm_val_eval_%x_%j.err

# Validation-set inference for sparse U-Net / MLP experiments.
# Pass --data-experiment (D2_alphaearth or A3_s2_9band) and one or more --exp names.
#
# Usage:
#   sbatch --job-name=ae_val_AE run_val_eval.slurm D2_alphaearth E4_ae_unet_sparse E4_D2_alphaearth_sparse_n5 ...
#   sbatch --job-name=ae_val_S2 run_val_eval.slurm A3_s2_9band E4_A3_s2_9band_sparse E4_rand_A3_s2_9band

DATA_EXP=${1:?"First arg: data-experiment (D2_alphaearth or A3_s2_9band)"}
shift
EXP_LIST="$@"

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

echo "=== Validation-set inference ==="
echo "Data experiment: ${DATA_EXP}"
echo "Experiments: ${EXP_LIST}"
echo ""

python experiments/exp3_annotation/scripts/modeling/evaluate_val_set.py \
    --data-experiment "${DATA_EXP}" \
    --exp ${EXP_LIST} \
    --num-folds 5

echo "=== Done ==="
