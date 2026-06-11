#!/bin/bash
#SBATCH --job-name=rotation_padding
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=preprocessing/outputs/rotation_padding_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

python preprocessing/scripts/06_analyze_rotation_padding.py
