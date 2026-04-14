#!/usr/bin/env python3
"""
Generate sparse point labels by sampling from dense masks.

Simulates a point-based annotation workflow: for each tile, randomly sample
N_POS positive (change) pixels and N_NEG negative (no-change) pixels.

Fallback: if a tile has fewer than N_POS positive pixels, sample all available
positive pixels and match with equal negatives.

Output: JSON file with per-tile (y, x, label) coordinates, reusable across
all Block E experiments and folds.
"""

import json
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "epsg3035_10m_v2"
MASK_DIR = PROCESSED_DIR / "masks"
MASK_PATTERN = "{refid}_mask.tif"
SPLITS_CSV = REPO_ROOT / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "sparse_labels"

DEFAULT_N_POS = 25
DEFAULT_N_NEG = 25
SEED = 42


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate sparse point labels")
    parser.add_argument("--n-pos", type=int, default=DEFAULT_N_POS, help="Target positive points per tile")
    parser.add_argument("--n-neg", type=int, default=DEFAULT_N_NEG, help="Target negative points per tile")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    N_POS = args.n_pos
    N_NEG = args.n_neg

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sparse_labels_n{N_POS}_seed{args.seed}.json"

    df = pd.read_csv(SPLITS_CSV)
    refids = df["refid"].tolist()
    print(f"Generating sparse labels for {len(refids)} tiles")
    print(f"Target: {N_POS} positive + {N_NEG} negative per tile")
    print(f"Seed: {args.seed}")

    rng = np.random.default_rng(args.seed)
    tiles = {}
    fallback_tiles = []
    total_points = 0

    for refid in refids:
        mask_path = MASK_DIR / MASK_PATTERN.format(refid=refid)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        pos_coords = np.argwhere(mask > 0)  # (N_pos, 2) — [y, x]
        neg_coords = np.argwhere(mask == 0)

        n_pos_available = len(pos_coords)
        n_pos = min(N_POS, n_pos_available)
        n_neg = n_pos  # match negatives to actual positives for balance

        if n_pos < N_POS:
            fallback_tiles.append(refid)

        # Sample
        pos_idx = rng.choice(n_pos_available, size=n_pos, replace=False)
        neg_idx = rng.choice(len(neg_coords), size=n_neg, replace=False)

        points = []
        for i in pos_idx:
            y, x = int(pos_coords[i, 0]), int(pos_coords[i, 1])
            points.append([y, x, 1])
        for i in neg_idx:
            y, x = int(neg_coords[i, 0]), int(neg_coords[i, 1])
            points.append([y, x, 0])

        tiles[refid] = points
        total_points += len(points)

    result = {
        "seed": SEED,
        "n_pos_target": N_POS,
        "n_neg_target": N_NEG,
        "tiles": tiles,
        "summary": {
            "total_tiles": len(tiles),
            "fallback_tiles": fallback_tiles,
            "total_points": total_points,
            "mean_points_per_tile": total_points / len(tiles),
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {out_path}")
    print(f"Total tiles: {len(tiles)}")
    print(f"Total points: {total_points} (mean {total_points/len(tiles):.1f}/tile)")
    print(f"Fallback tiles (<{N_POS} positive pixels): {len(fallback_tiles)}")
    for r in fallback_tiles:
        n = sum(1 for p in tiles[r] if p[2] == 1)
        print(f"  {r}: {n} positive (of {N_POS} target)")


if __name__ == "__main__":
    main()
