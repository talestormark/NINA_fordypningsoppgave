#!/usr/bin/env python3
"""
Train Random Forest on sparse point labels, evaluate per-tile IoU.

Supports two input types:
  - alphaearth: 64 features × 2 dates = 128 features/pixel (E1)
  - sentinel: 9 bands × 2 dates = 18 features/pixel (E2)

Uses 5-fold stratified CV with identical fold assignments as U-Net experiments.
Predicts ALL pixels per tile and computes IoU against full dense mask.
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "epsg3035_10m_v2"
SPLITS_CSV = REPO_ROOT / "preprocessing" / "outputs" / "splits" / "unified" / "split_info.csv"
STATS_JSON = PROCESSED_DIR / "normalisation_stats.json"
ANNOTATION_META_CSV = REPO_ROOT / "data_v2" / "annotations_metadata_final.csv"

MODALITY_DIRS = {
    "sentinel": PROCESSED_DIR / "sentinel",
    "alphaearth": PROCESSED_DIR / "alphaearth",
    "masks": PROCESSED_DIR / "masks",
}
MODALITY_PATTERNS = {
    "sentinel": "{refid}_RGBNIRRSWIRQ_Mosaic.tif",
    "alphaearth": "{refid}_VEY_Mosaic.tif",
    "masks": "{refid}_mask.tif",
}

S2_N_TIMESTEPS = 14
S2_N_BANDS = 9
AE_N_YEARS = 7
AE_N_FEATURES = 64
N_YEARS = 7
EPS = 1e-7


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_normalisation_stats():
    with open(STATS_JSON) as f:
        raw = json.load(f)
    stats = {}
    for key in ("sentinel", "alphaearth"):
        if key in raw:
            stats[key] = {
                "mean": np.array(raw[key]["mean"], dtype=np.float64),
                "std": np.array(raw[key]["std"], dtype=np.float64),
            }
    return stats


def load_annotation_years():
    """Load per-tile annotation years, clamped to S2 range [2018, 2024]."""
    meta = pd.read_csv(ANNOTATION_META_CSV)
    years = {}
    for _, row in meta.iterrows():
        if pd.isna(row["startYear"]) or pd.isna(row["endYear"]):
            continue
        start_idx = max(int(row["startYear"]), 2018) - 2018
        end_idx = min(int(row["endYear"]), 2024) - 2018
        years[row["REFID"]] = (start_idx, end_idx)
    return years


def _compose_annual(data):
    """Reduce 14 quarterly timesteps to 7 annual composites (Q2+Q3 average)."""
    composites = []
    for year_idx in range(N_YEARS):
        q2_idx = year_idx * 2
        q3_idx = year_idx * 2 + 1
        q2 = data[q2_idx]
        q3 = data[q3_idx]
        q2_nan_pct = np.isnan(q2).sum() / q2.size * 100
        q3_nan_pct = np.isnan(q3).sum() / q3.size * 100
        if q2_nan_pct > 50 and q3_nan_pct < 20:
            composites.append(q3)
        elif q3_nan_pct > 50 and q2_nan_pct < 20:
            composites.append(q2)
        else:
            composites.append((q2 + q3) / 2.0)
    return np.stack(composites, axis=0)


def load_tile_features(refid, input_type, norm_stats, tile_years, coords=None):
    """
    Load features for a tile. If coords is None, load ALL valid pixels.

    Returns:
        features: (N, D) array — D=128 for alphaearth, D=18 for sentinel
        valid_mask: (H, W) boolean — True for valid pixels (not border-fill)
        shape: (H, W) tuple
    """
    start_idx, end_idx = tile_years.get(refid, (0, 6))

    if input_type == "alphaearth":
        path = MODALITY_DIRS["alphaearth"] / MODALITY_PATTERNS["alphaearth"].format(refid=refid)
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float64)  # (448, H, W)
        data = raw.reshape(AE_N_YEARS, AE_N_FEATURES, raw.shape[1], raw.shape[2])
        data = data[[start_idx, end_idx]]  # (2, 64, H, W)
        # Valid pixels: not all-zero in raw data
        valid = np.any(raw != 0, axis=0)  # (H, W)
        # Normalize
        mean = norm_stats["alphaearth"]["mean"]
        std = norm_stats["alphaearth"]["std"]
        data = (data - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)

    elif input_type == "sentinel":
        path = MODALITY_DIRS["sentinel"] / MODALITY_PATTERNS["sentinel"].format(refid=refid)
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float64)  # (126, H, W)
        all_ts = raw.reshape(S2_N_TIMESTEPS, S2_N_BANDS, raw.shape[1], raw.shape[2])
        composites = _compose_annual(all_ts)  # (7, 9, H, W)
        data = composites[[start_idx, end_idx]]  # (2, 9, H, W)
        valid = np.any(raw != 0, axis=0)
        mean = norm_stats["sentinel"]["mean"]
        std = norm_stats["sentinel"]["std"]
        data = (data - mean[None, :, None, None]) / (std[None, :, None, None] + EPS)
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    np.nan_to_num(data, copy=False, nan=0.0)
    H, W = data.shape[2], data.shape[3]

    if coords is not None:
        # Extract features at specific coordinates
        features = []
        for y, x, _ in coords:
            feat = data[:, :, y, x].flatten()  # (T*C,)
            features.append(feat)
        return np.array(features), valid, (H, W)
    else:
        # Full tile: flatten all pixels
        # (2, C, H, W) → (H, W, 2, C) → (H*W, 2*C)
        features = data.transpose(2, 3, 0, 1).reshape(H * W, -1)
        return features, valid, (H, W)


def load_mask(refid):
    path = MODALITY_DIRS["masks"] / MODALITY_PATTERNS["masks"].format(refid=refid)
    with rasterio.open(path) as src:
        mask = src.read(1)
    return (mask > 0).astype(np.int32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_tile_metrics(pred_binary, mask_binary, valid):
    """Compute per-tile metrics on valid pixels only."""
    pred = pred_binary[valid]
    gt = mask_binary[valid]

    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    iou = tp / (tp + fp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Fold splitting (matches dataset.py exactly)
# ---------------------------------------------------------------------------

def get_fold_splits(fold, num_folds=5, seed=42):
    df = pd.read_csv(SPLITS_CSV)
    train_refids = df[df["split"] == "train"]["refid"].tolist()
    val_refids = df[df["split"] == "val"]["refid"].tolist()
    test_refids = df[df["split"] == "test"]["refid"].tolist()

    trainval_refids = train_refids + val_refids
    refid_to_level = dict(zip(df["refid"], df["change_level"]))
    change_levels = [refid_to_level[r] for r in trainval_refids]

    skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(skfold.split(trainval_refids, change_levels))

    train_idx, val_idx = splits[fold]
    fold_train = [trainval_refids[i] for i in train_idx]
    fold_val = [trainval_refids[i] for i in val_idx]

    return fold_train, fold_val, test_refids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fold(args, fold, sparse_labels, norm_stats, tile_years):
    """Train and evaluate one fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")

    train_refids, val_refids, test_refids = get_fold_splits(fold, args.num_folds, args.seed)

    # Optional training fraction
    if args.train_fraction < 1.0:
        rng = np.random.default_rng(args.seed + fold)
        n_keep = max(1, int(len(train_refids) * args.train_fraction))
        train_refids = sorted(train_refids)
        train_refids = list(rng.choice(train_refids, size=n_keep, replace=False))
        print(f"  Train fraction: {args.train_fraction} → {len(train_refids)} tiles")
    else:
        print(f"  Train tiles: {len(train_refids)}")

    print(f"  Val tiles: {len(val_refids)}")
    print(f"  Test tiles: {len(test_refids)}")

    # Build training matrix from sparse labels
    X_parts, y_parts = [], []
    skipped = 0
    for refid in train_refids:
        if refid not in sparse_labels["tiles"]:
            skipped += 1
            continue
        coords = sparse_labels["tiles"][refid]
        features, _, _ = load_tile_features(refid, args.input_type, norm_stats, tile_years, coords=coords)
        labels = np.array([c[2] for c in coords])
        X_parts.append(features)
        y_parts.append(labels)

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    print(f"  Training: {len(X_train)} points ({y_train.sum()} pos, {(1-y_train).sum():.0f} neg), "
          f"{X_train.shape[1]} features, {skipped} tiles skipped")

    # Train RF
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"  RF trained in {time.time()-t0:.1f}s")

    # Evaluate on val and test
    results = {"fold": fold, "input_type": args.input_type, "n_train_tiles": len(train_refids),
               "n_train_points": len(X_train), "n_estimators": args.n_estimators,
               "train_fraction": args.train_fraction}

    for split_name, split_refids in [("val", val_refids), ("test", test_refids)]:
        per_tile = {}
        all_probs, all_masks = [], []
        t0 = time.time()

        for refid in split_refids:
            features, valid, (H, W) = load_tile_features(refid, args.input_type, norm_stats, tile_years)
            mask = load_mask(refid)

            probs = rf.predict_proba(features)[:, 1]
            probs_map = probs.reshape(H, W)
            pred_binary = (probs_map > 0.5).astype(np.int32)

            metrics = compute_tile_metrics(pred_binary, mask, valid)
            per_tile[refid] = metrics

            # Accumulate for micro metrics
            v = valid.ravel()
            all_probs.append(probs[v])
            all_masks.append(mask.ravel()[v])

        # Macro metrics (per-tile average)
        ious = [m["iou"] for m in per_tile.values()]
        f1s = [m["f1"] for m in per_tile.values()]

        # Micro metrics (pooled)
        total_tp = sum(m["tp"] for m in per_tile.values())
        total_fp = sum(m["fp"] for m in per_tile.values())
        total_fn = sum(m["fn"] for m in per_tile.values())
        micro_iou = total_tp / (total_tp + total_fp + total_fn + EPS)
        micro_prec = total_tp / (total_tp + total_fp + EPS)
        micro_rec = total_tp / (total_tp + total_fn + EPS)
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + EPS)

        results[f"{split_name}_metrics"] = {
            "macro_iou": float(np.mean(ious)),
            "macro_iou_std": float(np.std(ious)),
            "macro_f1": float(np.mean(f1s)),
            "micro_iou": float(micro_iou),
            "micro_f1": float(micro_f1),
            "n_tiles": len(split_refids),
            "per_tile": per_tile,
        }
        print(f"  {split_name}: macro IoU={np.mean(ious)*100:.1f}% ± {np.std(ious)*100:.1f}%, "
              f"micro IoU={micro_iou*100:.1f}% ({time.time()-t0:.1f}s)")

    return results, rf


def main():
    parser = argparse.ArgumentParser(description="Train RF on sparse labels")
    parser.add_argument("--input-type", type=str, required=True, choices=["alphaearth", "sentinel"])
    parser.add_argument("--fold", type=int, default=-1, help="Fold (0-4), or -1 for all folds")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-labels", type=str, required=True, help="Path to sparse_labels JSON")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction of training tiles to use")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        base = Path(__file__).resolve().parents[2] / "outputs"
        if args.train_fraction < 1.0:
            pct = int(args.train_fraction * 100)
            exp_name = f"E1_ae_rf_sparse_{pct}pct" if args.input_type == "alphaearth" else f"E2_s2_rf_sparse_{pct}pct"
        else:
            exp_name = "E1_ae_rf_sparse" if args.input_type == "alphaearth" else "E2_s2_rf_sparse"
        args.output_dir = str(base / exp_name)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared resources
    print(f"Input type: {args.input_type}")
    print(f"Sparse labels: {args.sparse_labels}")
    print(f"Train fraction: {args.train_fraction}")

    with open(args.sparse_labels) as f:
        sparse_labels = json.load(f)
    print(f"Sparse labels: {sparse_labels['summary']['total_tiles']} tiles, "
          f"{sparse_labels['summary']['total_points']} points")

    norm_stats = load_normalisation_stats()
    tile_years = load_annotation_years()

    # Run folds
    folds = range(args.num_folds) if args.fold == -1 else [args.fold]
    all_results = []

    for fold in folds:
        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        results, rf = run_fold(args, fold, sparse_labels, norm_stats, tile_years)
        all_results.append(results)

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    # CV summary (if all folds)
    if len(all_results) == args.num_folds:
        val_ious = [r["val_metrics"]["macro_iou"] for r in all_results]
        test_ious = [r["test_metrics"]["macro_iou"] for r in all_results]

        summary = {
            "input_type": args.input_type,
            "n_estimators": args.n_estimators,
            "train_fraction": args.train_fraction,
            "cv_val_iou_mean": float(np.mean(val_ious)),
            "cv_val_iou_std": float(np.std(val_ious)),
            "cv_val_iou_folds": [float(x) for x in val_ious],
            "cv_test_iou_mean": float(np.mean(test_ious)),
            "cv_test_iou_std": float(np.std(test_ious)),
            "cv_test_iou_folds": [float(x) for x in test_ious],
        }

        with open(out_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("CV SUMMARY")
        print(f"{'='*60}")
        print(f"Val IoU:  {np.mean(val_ious)*100:.1f}% ± {np.std(val_ious)*100:.1f}%  "
              f"folds=[{', '.join(f'{x*100:.1f}' for x in val_ious)}]")
        print(f"Test IoU: {np.mean(test_ious)*100:.1f}% ± {np.std(test_ious)*100:.1f}%  "
              f"folds=[{', '.join(f'{x*100:.1f}' for x in test_ious)}]")


if __name__ == "__main__":
    main()
