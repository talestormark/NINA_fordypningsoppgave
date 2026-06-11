# Experiment 3 — Annotation efficiency (RQ3)

> *To what extent can sparse point-level annotations substitute for dense pixel-wise masks in
> land-take detection?*

This is the central question of the thesis. On the Experiment 1 baseline (EarlyFusion, `T=2`) and the
**260 tiles**, dense per-pixel masks are replaced by a few **sparse point labels** per tile, trained
with a **masked loss** (focal + dice restricted to the labelled pixels; unlabelled pixels do not
contribute to the gradient). Shared model/data/training code is imported from `landtake`; the
multi-source dataset is `landtake.data.spectral`.

## Sub-questions

- **RQ3a — sparse vs dense.** 50 balanced sparse points per tile vs the dense mask, on AlphaEarth.
- **RQ3b — input representation.** The same comparison on 9-band Sentinel-2 as well as AlphaEarth.
- **RQ3c — point budget.** Balanced budgets of 10, 20, 30, 50, 200 points per tile (AlphaEarth).
- **RQ3d — random placement.** Random vs balanced placement at 50 points, on three models: the
  AlphaEarth masked-loss U-Net, the Sentinel-2 masked-loss U-Net, and a per-pixel **Random Forest**
  baseline. (A per-pixel MLP control is reported in the thesis appendix.)

## Key findings

- **About 50 balanced points per tile reach dense-mask accuracy** (~1% of the labelled pixels); the
  boundary lies between 30 and 50 points, and going from 50 to 200 adds nothing detectable. This
  holds on both AlphaEarth and Sentinel-2.
- **The masked-loss U-Net is robust to simple random placement**, losing < 2 pp, whereas the Random
  Forest collapses (~16 pp) because tiles with zero positive labels give it no positive samples — so
  the end-to-end gradient training, not the sampling scheme, is what makes sparse supervision work.

## How to run

```bash
conda activate masterthesis && pip install -e .          # from the repo root, once

# 1. Generate sparse point labels (balanced 25+25; --random --n-total 50 for random;
#    --n-pos/--n-neg set the budget, e.g. 5+5, 10+10, 15+15, 25+25, 100+100)
sbatch experiments/exp3_annotation/scripts/slurm/generate_labels.slurm

# 2. Train the masked-loss U-Net: <data-experiment> <fold> <labels-file>
#    (data-experiment = D2_alphaearth or A3_s2_9band)
sbatch experiments/exp3_annotation/scripts/slurm/train_masked_unet.slurm D2_alphaearth 0 sparse_labels_seed42.json
# Random Forest baseline (RQ3d)
sbatch experiments/exp3_annotation/scripts/slurm/train_rf_e1.slurm

# 3. Held-out test evaluation (budget / random variants) + statistical tests
sbatch experiments/exp3_annotation/scripts/slurm/run_test_eval_budget.slurm
sbatch experiments/exp3_annotation/scripts/slurm/statistical_tests.slurm
```

Uses AlphaEarth / 9-band Sentinel-2 inputs and the 260-tile unified splits
(`preprocessing/outputs/splits/unified/`, 181 training tiles). Outputs are git-ignored.

## Layout

```
scripts/data_preparation/  generate_sparse_labels.py (budgets, balanced/random placement)
scripts/modeling/          train_masked_unet.py, train_masked_mlp.py, train_rf.py, evaluate_*.py
scripts/analysis/          budget curves, spatial-context / random-placement / loss-ablation stats
scripts/slurm/             label generation, training, evaluation, statistical tests
```
