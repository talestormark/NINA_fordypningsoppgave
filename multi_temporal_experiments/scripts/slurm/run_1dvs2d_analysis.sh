#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=1dvs2d_stats
#SBATCH --time=0-00:15:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# 1D vs 2D Temporal Modeling Statistical Analysis (RQ2f)
# ============================================================================

echo "=========================================="
echo "1D vs 2D STATISTICAL ANALYSIS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python multi_temporal_experiments/scripts/modeling/statistical_analysis_1dvs2d.py \
    --num-folds 5 \
    --n-permutations 10000 \
    --n-bootstrap 10000 \
    --seed 42

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
