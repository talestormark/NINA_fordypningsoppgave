#!/bin/bash
#SBATCH --job-name=pred_bf
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=PART2_spectral_spatial_resolution_experiments/outputs/logs/save_predictions_%j.log

# Step 1: Save ensemble test predictions for all experiments
# Step 2: Run boundary metrics (BF@1, BF@2) on saved predictions

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

ANALYSIS_DIR="PART2_spectral_spatial_resolution_experiments/scripts/analysis"

echo "========================================"
echo "Step 1: Saving test predictions"
echo "========================================"

python ${ANALYSIS_DIR}/save_test_predictions.py --all

echo ""
echo "========================================"
echo "Step 2: Running boundary metrics"
echo "========================================"

EXPERIMENTS_DIR="PART2_spectral_spatial_resolution_experiments/outputs/experiments"

for exp in A1_s2_rgb A2_s2_rgbnir A3_s2_9band A4_s2_indices D2_alphaearth E4_ae_unet_sparse E4_A3_s2_9band_sparse; do
    PRED_DIR="${EXPERIMENTS_DIR}/${exp}/predictions"
    if [ -d "$PRED_DIR" ]; then
        echo ""
        echo "--- Boundary metrics: ${exp} ---"
        python ${ANALYSIS_DIR}/boundary_metrics.py \
            --predictions-dir ${PRED_DIR} \
            --output ${PRED_DIR}/boundary_metrics.json
    else
        echo "WARNING: No predictions for ${exp}"
    fi
done

echo ""
echo "=== All done ==="
