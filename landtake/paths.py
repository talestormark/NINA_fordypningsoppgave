"""Canonical filesystem anchors for the repository.

``landtake/`` lives at the repository root, so ``REPO_ROOT`` is derived from this
file's location and stays correct under an editable install (``pip install -e .``).
Scripts should import these anchors instead of computing
``Path(__file__).resolve().parents[N]`` so that moving a file never breaks its
paths.
"""

from pathlib import Path

# landtake/paths.py -> parents[0] = landtake/, parents[1] = repository root
REPO_ROOT = Path(__file__).resolve().parents[1]

# Working data lives in data_v2/ (git-ignored). The training scripts take the
# data directory via --data-dir; landtake.config.DATA_DIR is the legacy default.
DATA_V2_DIR = REPO_ROOT / "data_v2"

# Shared preprocessing outputs (train/val/test splits, normalization stats).
SPLITS_DIR = REPO_ROOT / "preprocessing" / "outputs" / "splits"

# The three experiments live under experiments/.
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
