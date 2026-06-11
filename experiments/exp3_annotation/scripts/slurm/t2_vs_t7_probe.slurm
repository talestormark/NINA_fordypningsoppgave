#!/bin/bash
#SBATCH --job-name=ae_t2_vs_t7
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=experiments/exp3_annotation/outputs/linear_probe/slurm_t2vst7_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

python experiments/exp3_annotation/scripts/alphaearth_t2_vs_t7_probe.py
