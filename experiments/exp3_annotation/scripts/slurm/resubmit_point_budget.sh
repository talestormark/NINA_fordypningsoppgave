#!/bin/bash
# Resubmit point budget sweep with 6h time limit (was 3h, caused timeouts at ~200/400 epochs)
# Retrains from scratch — will overwrite best_model.pth with the full-training result
#
# Convergence analysis showed:
#   n5:   1/5 folds still improving at timeout (fold4)
#   n10:  1/5 folds still improving at timeout (fold4)
#   n100: 2/5 folds still improving at timeout (fold2, fold4)
#
# Usage: bash resubmit_point_budget.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/train_masked_unet.slurm"
AE_OUTPUTS="/cluster/home/tmstorma/NINA_fordypningsoppgave/experiments/exp3_annotation/outputs"

echo "Submitting point budget sweep with 6h time limit..."
echo ""

for VARIANT in n5 n10 n100; do
    LABELS_FILE="sparse_labels_${VARIANT}_seed42.json"
    OUTPUT_BASE="${AE_OUTPUTS}/E4_D2_alphaearth_sparse_${VARIANT}"

    for FOLD in 0 1 2 3 4; do
        OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
        JOB_NAME="ae_E4${VARIANT}_f${FOLD}"

        echo "Submitting ${JOB_NAME}: D2_alphaearth fold=${FOLD} labels=${LABELS_FILE}"

        sbatch \
            --job-name="${JOB_NAME}" \
            --time=0-06:00:00 \
            --output="${OUTPUT_BASE}/slurm_f${FOLD}_%j.log" \
            "${SLURM_TEMPLATE}" \
            D2_alphaearth "${FOLD}" "${LABELS_FILE}" "${OUTPUT_DIR}"
    done
    echo ""
done

echo "All jobs submitted. Monitor with: squeue -u \$USER"
