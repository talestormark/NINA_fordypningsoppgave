#!/bin/bash
#SBATCH --job-name=ae_stats
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=experiments/annotation_efficiency/outputs/statistical_tests/slurm_%j.log

module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave
python experiments/annotation_efficiency/scripts/analysis/statistical_tests.py
