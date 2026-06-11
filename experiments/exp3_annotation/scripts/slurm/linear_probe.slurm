#!/bin/bash
#SBATCH --job-name=ae_linear_probe
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=experiments/annotation_efficiency/outputs/linear_probe/slurm_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

python experiments/annotation_efficiency/scripts/alphaearth_linear_probe.py
