# Experiment 1 — Temporal sampling and fusion (RQ1)

> *How do temporal sampling density and the model's strategy for combining observations across time
> affect land-take detection from Sentinel-2 imagery?*

This experiment establishes the **baseline** carried into Experiments 2 and 3. The input is the
9-band Sentinel-2 stack (10 m, EPSG:3035) on the **163 full-window tiles**; either the number of
timesteps or the architecture is varied while everything else is held fixed. Shared model/data/
training code is imported from the `landtake` package; this directory holds only this experiment's
scripts.

## Sub-questions

- **RQ1a — temporal sampling density.** Bi-temporal (`T=2`, start + end year), annual (`T=7`),
  bi-seasonal (`T=14`), evaluated with the 2-layer ConvLSTM held fixed (the one architecture that
  spans all three `T`).
- **RQ1b — fusion strategy.** With `T` held fixed, eight architectures grouped by `T`:
  EarlyFusion, Concat, LSTM-2, LSTM-2-lite (`T=2`); Pool-7, Conv3D-7, LSTM-7, LSTM-7-lite (`T=7`).

## Key findings

- **Temporal density beyond two timesteps does not help:** `T=2` and `T=7` are statistically
  indistinguishable, and bi-seasonal `T=14` is slightly but significantly worse than annual.
- **No architecture is significantly better than another** at the same `T`; the parameter-matched
  *lite* ConvLSTMs match the full ones.
- The baseline chosen by parsimony is **EarlyFusion at `T=2`** (channel-stacked U-Net) — the simplest
  configuration that performs comparably — and it is used in Experiments 2 and 3.

## How to run

```bash
conda activate masterthesis && pip install -e .          # from the repo root, once

# 5-fold CV training; the launcher takes <experiment> <fold>.
# Experiment keys exp001..exp010 map to architectures in scripts/experiments_v2.py
# (exp001 annual LSTM-7, exp003 bi-temporal LSTM-2, exp005 EarlyFusion, exp006 Concat,
#  exp007 Pool-7, exp008 Conv3D-7, exp009/010 lite ConvLSTMs, exp002 bi-seasonal LSTM-14).
for exp in exp001 exp002 exp003 exp005 exp006 exp007 exp008 exp009 exp010; do
  for fold in 0 1 2 3 4; do
    sbatch experiments/exp1_temporal_sampling/scripts/slurm/v2/train_all_experiments_v3.sh $exp $fold
  done
done

# Held-out test evaluation, then the statistical-analysis pipeline
sbatch experiments/exp1_temporal_sampling/scripts/slurm/evaluate_test_per_fold.sh
sbatch experiments/exp1_temporal_sampling/scripts/slurm/v2/run_analysis_pipeline.sh
```

Training uses `data_v2`, masks `Land_take_masks_coarse/`, and the 5-fold splits in
`preprocessing/outputs/splits/unified_fullwindow/` (stratified by change level). Outputs
(checkpoints, metrics, logs) are written under `outputs_v3/` and are git-ignored.

## Layout

```
scripts/experiments_v2.py    experiment registry (exp00N -> architecture, paths, comparison families)
scripts/modeling/            train_multitemporal.py, evaluate_*.py, statistical_*.py, boundary_*.py
scripts/data_preparation/    Sentinel-2 temporal validation + per-fold normalisation stats
scripts/analysis/            per-fold / qualitative / temporal-importance analysis, training curves
scripts/slurm/               v2/ = current launchers (the v1/ autumn pilots are not published)
```
