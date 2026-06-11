#!/bin/bash
# Submit MLP under uniform random sparse placement (50 points/tile, AlphaEarth).
# Caveat 1 control experiment for Discussion §2: tests whether the per-pixel MLP
# survives random placement, mirroring the U-Net (E4-random). Existing E5 balanced
# was tested at 50 balanced points/tile.
#
# Usage: bash submit_mlp_random.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/train_masked_mlp.slurm"
AE_OUTPUTS="/cluster/home/tmstorma/NINA_fordypningsoppgave/experiments/exp3_annotation/outputs"

EXPERIMENT="D2_alphaearth"
LABELS_FILE="sparse_labels_random_n50_seed42.json"
OUTPUT_BASE="${AE_OUTPUTS}/E5_D2_alphaearth_mlp_sparse_random"

mkdir -p "${OUTPUT_BASE}"

echo "Submitting MLP-random (caveat 1 control)..."
echo "  Experiment: ${EXPERIMENT}"
echo "  Labels: ${LABELS_FILE}"
echo "  Output base: ${OUTPUT_BASE}"
echo ""

for FOLD in 0 1 2 3 4; do
    OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    JOB_NAME="ae_E5rand_f${FOLD}"

    echo "Submitting ${JOB_NAME}: fold=${FOLD}"

    sbatch \
        --job-name="${JOB_NAME}" \
        --time=0-06:00:00 \
        --output="${OUTPUT_BASE}/slurm_f${FOLD}_%j.log" \
        "${SLURM_TEMPLATE}" \
        "${EXPERIMENT}" "${FOLD}" "${LABELS_FILE}" "${OUTPUT_DIR}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
