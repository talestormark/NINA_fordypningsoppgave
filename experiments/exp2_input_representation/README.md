# Experiment 2 — Input representation (RQ2)

> *How does the choice of input representation affect land-take detection accuracy?*

On top of the Experiment 1 baseline (EarlyFusion, `T=2`), this experiment varies only the **input
representation** of the same **260 tiles** (clamped configuration), holding the architecture and
training protocol fixed. Shared model/data/training code is imported from the `landtake` package; the
multi-source dataset and its configuration keys live in `landtake.data.spectral` (`EXPERIMENT_CONFIGS`).

## Sub-questions

- **RQ2a — Sentinel-2 spectral subset.** Four subsets of the 9-band stack:
  RGB (`A1_s2_rgb`), RGB+NIR (`A2_s2_rgbnir`), the full 9-band reference (`A3_s2_9band`), and
  9-band + four spectral indices NDVI/NDBI/BSI/NDWI (`A4_s2_indices`).
- **RQ2b — foundation-model embeddings.** AlphaEarth (start + end year, 64×2 = 128 channels,
  `D2_alphaearth`) vs the 9-band Sentinel-2 reference, at full data **and** at reduced training-set
  sizes (9, 18, 35, 70, and the full ~176 tiles per fold).

(`landtake.data.spectral.EXPERIMENT_CONFIGS` also contains additional exploratory configurations
that are not reported in the thesis.)

## Key findings

- **RGB+NIR is as accurate as the full 9-band stack**; RGB alone is borderline worse; adding the
  four spectral indices does not improve accuracy. No band comparison is statistically significant.
- **AlphaEarth does no better than raw 9-band Sentinel-2** — at full data or at any smaller
  training-set size (it shows no low-data advantage down to 9 tiles per fold).

## How to run

```bash
conda activate masterthesis && pip install -e .          # from the repo root, once

# RQ2a — train each spectral subset across folds, then evaluate on the held-out test set
for EXP in A1_s2_rgb A2_s2_rgbnir A3_s2_9band A4_s2_indices D2_alphaearth; do
  for f in 0 1 2 3 4; do
    sbatch --job-name=p2_${EXP}_f${f} experiments/exp2_input_representation/scripts/slurm/train_experiment.sh $EXP $f
  done
done
sbatch experiments/exp2_input_representation/scripts/slurm/evaluate_experiment.sh A3_s2_9band

# RQ2b — data-efficiency sweep ({A3_s2_9band, D2_alphaearth} x {9,18,35,70 tiles} x folds)
bash experiments/exp2_input_representation/scripts/slurm/submit_data_efficiency_sweep.sh
```

Training uses the 260-tile unified splits (`preprocessing/outputs/splits/unified/`, 5-fold stratified
by change level) over the reprojected data under `data/processed/epsg3035_10m_v2/`. Outputs are
written under `outputs/` and are git-ignored.

## Layout

```
scripts/modeling/      train.py (config-driven), evaluate_*.py
scripts/preprocessing/ EPSG:3035 reprojection + new-data verification
scripts/analysis/      stratified / per-sample / boundary / per-class analysis, statistical tests
scripts/figures/       result plots, condition overviews, reprojection comparisons
scripts/slurm/         training, evaluation, and data-efficiency-sweep launchers
```
