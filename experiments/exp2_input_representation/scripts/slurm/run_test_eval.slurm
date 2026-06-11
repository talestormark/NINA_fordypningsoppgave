#!/bin/bash
#SBATCH --job-name=p2_test_eval
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=PART2_spectral_spatial_resolution_experiments/outputs/logs/test_eval_%j.log

module --quiet purge
module load Anaconda3/2024.02-1
conda activate masterthesis

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

for exp in A3_s2_9band A1_s2_rgb A2_s2_rgbnir A4_s2_indices D2_alphaearth E4_ae_unet_sparse; do
    echo ""
    echo "========================================"
    echo "Test evaluation: $exp"
    echo "========================================"
    python PART2_spectral_spatial_resolution_experiments/scripts/modeling/evaluate_test_set.py \
        --experiment $exp --all-folds
done

echo ""
echo "=== All test evaluations complete ==="
