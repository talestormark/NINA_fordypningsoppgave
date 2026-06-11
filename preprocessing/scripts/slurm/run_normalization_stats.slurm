#!/bin/bash
#SBATCH --job-name=norm_stats
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=preprocessing/outputs/normalization_stats_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

python preprocessing/scripts/07_compute_normalization_stats.py
