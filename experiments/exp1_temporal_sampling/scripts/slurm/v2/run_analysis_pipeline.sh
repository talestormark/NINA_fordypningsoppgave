#!/bin/bash
# ============================================================================
# Analysis Pipeline Orchestrator for Part 1 v2
# ============================================================================
#
# Launches all analysis jobs with proper dependency chains.
#
# Usage:
#   bash run_analysis_pipeline.sh --phase1-only   # After exp001-003 complete
#   bash run_analysis_pipeline.sh --full           # After all 50 jobs complete
#
# Phase 1: Only RQ1 experiments (annual, bi_temporal, bi_seasonal)
# Full: All 10 experiments
#
# ============================================================================

set -euo pipefail

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

# Parse arguments
MODE="${1:---full}"

case "$MODE" in
    "--phase1-only"|"--phase1")
        EXPERIMENTS="annual,bi_temporal,bi_seasonal"
        PHASE="phase1"
        echo "=== PHASE 1 ANALYSIS (RQ1 only: exp001-003) ==="
        ;;
    "--full")
        EXPERIMENTS="annual,bi_temporal,bi_seasonal,early_fusion,late_fusion,late_fusion_pool,conv3d_fusion,lstm_lite,lstm7_lite"
        PHASE="full"
        echo "=== FULL ANALYSIS (9 trained experiments) ==="
        ;;
    *)
        echo "Usage: $0 [--phase1-only|--full]"
        exit 1
        ;;
esac

# Common SBATCH settings
ACCOUNT="share-ie-idi"
PARTITION="GPUQ"
LOG_DIR="experiments/exp1_temporal_sampling/outputs_v2/logs"
mkdir -p "$LOG_DIR"

CONDA_SETUP="module --quiet purge && module load Anaconda3/2024.02-1 && source activate masterthesis"
PROJECT_DIR="/cluster/home/tmstorma/NINA_fordypningsoppgave"

# Build experiments flag
if [ -n "$EXPERIMENTS" ]; then
    EXP_FLAG="--experiments $EXPERIMENTS"
else
    EXP_FLAG=""
fi

echo ""
echo "Experiments: ${EXPERIMENTS:-all}"
echo "Log directory: $LOG_DIR"
echo ""

# ============================================================================
# Tier 0: Sanity check (CPU, no dependencies)
# ============================================================================

echo "--- Tier 0: Sanity checks ---"

# Check that all expected checkpoints exist
SANITY_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=CPUQ \
    --job-name=sanity_check_${PHASE} \
    --time=0-00:10:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --output=$LOG_DIR/sanity_check_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python -c \"
from experiments/exp1_temporal_sampling.scripts.experiments_v2 import check_all_completeness, EXPERIMENTS_V2
status = check_all_completeness()
all_ok = True
exps = '${EXPERIMENTS}'.split(',') if '${EXPERIMENTS}' else list(EXPERIMENTS_V2.keys())
for key in exps:
    s = status[key]
    if s['complete']:
        print(f'  {key}: OK (all 5 folds)')
    else:
        print(f'  {key}: MISSING folds {s[\"missing\"]}')
        all_ok = False
if not all_ok:
    raise RuntimeError('Some experiments are incomplete!')
print('All checkpoints present.')
\"")
echo "  Sanity check: job $SANITY_JOB"

# ============================================================================
# Tier 1: GPU analysis jobs (parallel, depend on sanity check)
# ============================================================================

echo ""
echo "--- Tier 1: GPU analysis jobs ---"

# A. Per-sample statistical analysis
PERSAMPLE_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=persample_${PHASE} \
    --time=0-04:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/persample_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/modeling/statistical_analysis_persample.py $EXP_FLAG")
echo "  Per-sample IoU: job $PERSAMPLE_JOB"

# B. Boundary F-score analysis
BOUNDARY_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=boundary_${PHASE} \
    --time=0-04:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/boundary_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/modeling/boundary_f_score_analysis.py $EXP_FLAG")
echo "  Boundary F-score: job $BOUNDARY_JOB"

# C. Test set evaluation
TEST_EVAL_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=test_eval_${PHASE} \
    --time=0-02:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/test_eval_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/modeling/evaluate_test_final.py ${EXP_FLAG:-'--all-conditions'}")
echo "  Test evaluation: job $TEST_EVAL_JOB"

# D. Calibration & PR curves
CALIBRATION_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=calibration_${PHASE} \
    --time=0-04:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/calibration_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/calibration_pr_analysis.py $EXP_FLAG")
echo "  Calibration & PR: job $CALIBRATION_JOB"

# E. Qualitative CV analysis (depends on per-sample IoU)
QUAL_CV_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=qual_cv_${PHASE} \
    --time=0-01:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$PERSAMPLE_JOB \
    --output=$LOG_DIR/qual_cv_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/qualitative_cv_analysis.py --num-examples 3")
echo "  Qualitative CV: job $QUAL_CV_JOB (after persample)"

# F. Qualitative test analysis
QUAL_TEST_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=qual_test_${PHASE} \
    --time=0-01:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/qual_test_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/qualitative_test_analysis.py")
echo "  Qualitative test: job $QUAL_TEST_JOB"

# G. Stratified by change type
STRATIFIED_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --gres=gpu:1 \
    --constraint=gpu80g \
    --job-name=stratified_${PHASE} \
    --time=0-02:00:00 \
    --mem=64G \
    --cpus-per-task=4 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/stratified_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/stratified_by_change_type.py")
echo "  Stratified analysis: job $STRATIFIED_JOB"

# ============================================================================
# Tier 2: CPU analysis jobs (depend on Tier 1 GPU jobs)
# ============================================================================

echo ""
echo "--- Tier 2: CPU analysis jobs (after GPU jobs) ---"

# H. IoU distribution plots (depends on per-sample IoU)
PLOTS_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=CPUQ \
    --job-name=plots_${PHASE} \
    --time=0-00:30:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --dependency=afterok:$PERSAMPLE_JOB \
    --output=$LOG_DIR/plots_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/plot_iou_distributions.py")
echo "  IoU distribution plots: job $PLOTS_JOB (after persample)"

# I. Training curves (depends only on history.json, so after sanity check)
CURVES_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=CPUQ \
    --job-name=curves_${PHASE} \
    --time=0-00:30:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/curves_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/plot_training_curves.py")
echo "  Training curves: job $CURVES_JOB"

# J. Fold-level statistical analysis (depends only on history.json)
STATS_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --partition=CPUQ \
    --job-name=stats_${PHASE} \
    --time=0-00:30:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --dependency=afterok:$SANITY_JOB \
    --output=$LOG_DIR/stats_%j.log \
    --wrap="$CONDA_SETUP && cd $PROJECT_DIR && python experiments/exp1_temporal_sampling/scripts/analysis/statistical_analysis.py")
echo "  Fold-level stats: job $STATS_JOB"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "ANALYSIS PIPELINE SUBMITTED ($PHASE)"
echo "============================================"
echo ""
echo "Tier 0 (sanity):  $SANITY_JOB"
echo "Tier 1 (GPU):     $PERSAMPLE_JOB $BOUNDARY_JOB $TEST_EVAL_JOB $CALIBRATION_JOB $QUAL_CV_JOB $QUAL_TEST_JOB $STRATIFIED_JOB"
echo "Tier 2 (CPU):     $PLOTS_JOB $CURVES_JOB $STATS_JOB"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: experiments/exp1_temporal_sampling/outputs_v2/analysis/"
echo ""
