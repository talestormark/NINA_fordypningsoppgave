#!/bin/bash
#SBATCH --job-name=ae_gen_labels
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=experiments/exp3_annotation/outputs/sparse_labels/slurm_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave
python experiments/exp3_annotation/scripts/data_preparation/generate_sparse_labels.py
