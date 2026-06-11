#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp3_annotation/outputs/E4_ae_unet_sparse/slurm_%x_%j.log

# E4: Masked-loss U-Net on AlphaEarth with sparse labels
# Usage: sbatch --job-name=ae_E4_f0 train_e4.slurm 0

FOLD=${1:?"Usage: sbatch train_e4.slurm FOLD"}

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

LABELS=experiments/exp3_annotation/outputs/sparse_labels/sparse_labels_seed42.json

echo "=== E4: Masked-loss U-Net, fold ${FOLD} ==="
python experiments/exp3_annotation/scripts/modeling/train_masked_unet.py \
    --fold $FOLD \
    --sparse-labels $LABELS \
    --epochs 400 \
    --lr 0.01 \
    --batch-size 8 \
    --seed 42

echo "=== Done ==="
