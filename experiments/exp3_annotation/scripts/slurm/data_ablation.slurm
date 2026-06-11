#!/bin/bash
#SBATCH --job-name=ae_ablation
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=experiments/annotation_efficiency/outputs/data_ablation/slurm_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

LABELS=experiments/annotation_efficiency/outputs/sparse_labels/sparse_labels_seed42.json

for frac in 0.25 0.50 0.75; do
    echo ""
    echo "============================================"
    echo "Data ablation: ${frac} of training tiles"
    echo "============================================"
    python experiments/annotation_efficiency/scripts/modeling/train_rf.py \
        --input-type alphaearth \
        --fold -1 \
        --n-estimators 500 \
        --sparse-labels $LABELS \
        --train-fraction $frac
done
