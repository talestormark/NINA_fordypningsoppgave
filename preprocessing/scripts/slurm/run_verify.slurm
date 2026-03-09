#!/bin/bash
#SBATCH --job-name=verify_data_v2
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=preprocessing/outputs/verify_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

python preprocessing/scripts/02_verify_data.py --verbose
