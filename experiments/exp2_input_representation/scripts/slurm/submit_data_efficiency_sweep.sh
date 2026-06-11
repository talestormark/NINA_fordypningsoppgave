#!/bin/bash
# ============================================================================
# RQ2b data-efficiency sweep: AlphaEarth vs Sentinel-2 as training data shrinks
# ============================================================================
# Submits {A3_s2_9band, D2_alphaearth} x {9,18,35,70 tiles} x {folds 0-4} = 40 jobs.
# The full-data (176 tiles/fold) anchor reuses the EXISTING A3_s2_9band /
# D2_alphaearth runs, so it is not re-trained here.
#
# Each job reuses train_experiment.sh with the optional 3rd positional
# OUTPUT_NAME and an extra --train-subset N flag:
#   train_experiment.sh <EXP> <FOLD> <EXP>_n<N> --train-subset <N>
#   -> output dir: outputs/experiments/<EXP>_n<N>_fold<FOLD>
#
# Usage (from repo root):
#   bash experiments/exp2_input_representation/scripts/slurm/submit_data_efficiency_sweep.sh
# ============================================================================

cd /cluster/home/tmstorma/NINA_fordypningsoppgave

SLURM=experiments/exp2_input_representation/scripts/slurm/train_experiment.sh
EXPERIMENTS=(A3_s2_9band D2_alphaearth)
SIZES=(9 18 35 70)
FOLDS=(0 1 2 3 4)

n=0
for EXP in "${EXPERIMENTS[@]}"; do
  for N in "${SIZES[@]}"; do
    for F in "${FOLDS[@]}"; do
      sbatch --job-name="p2_${EXP}_n${N}_f${F}" \
        "$SLURM" "$EXP" "$F" "${EXP}_n${N}" --train-subset "$N"
      n=$((n+1))
    done
  done
done
echo "Submitted $n jobs."
