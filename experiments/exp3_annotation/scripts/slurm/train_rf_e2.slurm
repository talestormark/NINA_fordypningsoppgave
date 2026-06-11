#!/bin/bash
#SBATCH --job-name=ae_E2_rf
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=experiments/annotation_efficiency/outputs/E2_s2_rf_sparse/slurm_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

python experiments/annotation_efficiency/scripts/modeling/train_rf.py \
    --input-type sentinel \
    --fold -1 \
    --n-estimators 500 \
    --sparse-labels experiments/annotation_efficiency/outputs/sparse_labels/sparse_labels_seed42.json
