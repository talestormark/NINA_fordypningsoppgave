#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/exp2_input_representation/outputs/logs/slurm_%x_%j.log
#SBATCH --error=experiments/exp2_input_representation/outputs/logs/slurm_%x_%j.err
# ============================================================================
# Per-tile macro evaluation for the RQ2b data-efficiency sweep.
# Runs evaluate_val_set.py (CV macro) and evaluate_test_set.py (test ensemble)
# for each {A3_s2_9band, D2_alphaearth} x {9,18,35,70}. The data config is read
# from each run's config.json, so passing --experiment <EXP>_n<N> only selects
# the checkpoint dir. Writes <EXP>_n<N>/cv_macro_summary.json and test_results.json.
#
# Submit AFTER the training sweep finishes:
#   sbatch --job-name=p2_de_eval eval_data_efficiency_sweep.sh
# ============================================================================
cd /cluster/home/tmstorma/NINA_fordypningsoppgave
module --quiet purge
module load Anaconda3/2024.02-1
source activate masterthesis

MOD=experiments/exp2_input_representation/scripts/modeling
for EXP in A3_s2_9band D2_alphaearth; do
  for N in 9 18 35 70; do
    echo "==================== ${EXP}_n${N} ===================="
    python $MOD/evaluate_val_set.py  --experiment ${EXP}_n${N} --all-folds
    python $MOD/evaluate_test_set.py --experiment ${EXP}_n${N} --all-folds
  done
done
echo "Done at $(date)"
