#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=stats_analysis
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.log
#SBATCH --error=PART1_multi_temporal_experiments/outputs/logs/slurm_%x_%j.err

# ============================================================================
# Per-Sample Statistical Analysis
# ============================================================================
#
# Computes per-sample IoU from out-of-fold predictions (n=45 tiles) and
# performs rigorous statistical comparisons between temporal sampling conditions.
#
# Statistical tests:
# - Wilcoxon signed-rank test (paired, nonparametric)
# - Permutation test (10,000 permutations)
# - Bootstrap 95% CI for mean difference
# - Cliff's delta effect size
# - Holm-Bonferroni correction for multiple comparisons
#
# ============================================================================

echo "=========================================="
echo "PER-SAMPLE STATISTICAL ANALYSIS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "=========================================="

# Navigate to project root
cd /cluster/home/tmstorma/NINA_fordypningsoppgave

# Load environment
module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run per-sample statistical analysis (IoU)
echo "--- Per-sample statistical analysis ---"
python PART1_multi_temporal_experiments/scripts/modeling/statistical_analysis_persample.py \
    --num-folds 5 \
    --n-permutations 10000 \
    --n-bootstrap 10000 \
    --seed 42

echo ""
echo "--- Boundary F-score analysis ---"
python PART1_multi_temporal_experiments/scripts/modeling/boundary_f_score_analysis.py \
    --tolerance 2 \
    --threshold 0.5 \
    --num-folds 5 \
    --n-permutations 10000 \
    --seed 42

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
