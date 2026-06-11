# Deep Learning for Land-Take Detection from Sentinel-2

Code for the master's thesis *Deep Learning for Land-Take Detection from Sentinel-2
Satellite Imagery* (NTNU, in collaboration with NINA). **Land take** is the conversion of
non-artificial cover to artificial cover. The thesis frames its detection as binary change
detection / semantic segmentation at 10 m over Europe, and studies how much **temporal,
spectral, and annotation detail** is actually needed.

The repository is organised as a shared library plus three parallel experiments, mirroring
the methodology pipeline (data preparation → model training → inference → evaluation, with an
across-experiment comparison).

## Structure

```
landtake/          Installable shared library — every experiment imports from here:
                     models (multitemporal + ConvLSTM), losses, metrics, logger,
                     datasets (multitemporal Sentinel-2, multi-source spectral),
                     and canonical paths/config (no sys.path hacks).
preprocessing/     Data preparation: reprojection, splits, normalisation, mask analysis.
experiments/
  exp1_temporal_sampling/      RQ1 — temporal sampling density and fusion
  exp2_input_representation/   RQ2 — spectral bands and AlphaEarth embeddings
  exp3_annotation/             RQ3 — sparse point annotation
analysis/          Across-experiment comparison (paired permutation tests, effect sizes,
                   report-number verification).
figures/           Thesis-figure generators.
tools/             smoke.py (data-free) and data_smoke.py (1-tile end-to-end) checks.
pyproject.toml     Editable install of the landtake package.
environment.yml    Conda environment specification.
```

Each experiment directory holds only what is specific to it (its training/evaluation/analysis
scripts and configuration) and imports the shared machinery from `landtake`.

## Installation

```bash
conda env create -f environment.yml
conda activate masterthesis
pip install -e .          # exposes the `landtake` package on the import path
```

`pip install -e .` is what replaces the old per-script path manipulation: scripts simply
`from landtake.… import …`.

## Quick checks

```bash
python tools/smoke.py        # imports + build a model + loss on random tensors (no data, no GPU)
python tools/data_smoke.py   # one real tile through each input pathway (needs data present)
```

## Reproducing the experiments

Large data and outputs are git-ignored (see below), so the steps assume the data is in place:

1. **Data preparation** — `preprocessing/scripts/` (verify data, build train/val/test splits,
   compute normalisation statistics, analyse masks).
2. **Training / evaluation** — each experiment's `scripts/modeling/` (launched on the cluster via
   the SLURM scripts under each experiment's `scripts/slurm/`).
3. **Figures and statistics** — `figures/` (thesis figures) and `analysis/` (across-experiment
   paired tests and effect sizes).

## Data and validation

The repository contains **code only**. The imagery (Sentinel-2, AlphaEarth, PlanetScope, VHR),
land-take masks, model checkpoints, experiment outputs and logs are git-ignored.

Data validation lives in the kept pipeline: `preprocessing/scripts/02_verify_data.py`
(CRS / pixel size / band counts / grid alignment for all sources) and
`experiments/exp1_temporal_sampling/scripts/data_preparation/01_validate_sentinel2_temporal.py`
(per-quarter Sentinel-2 quality).

## Author

Tale Stormark — NTNU master's thesis, in collaboration with NINA.
