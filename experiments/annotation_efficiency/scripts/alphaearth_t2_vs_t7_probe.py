#!/usr/bin/env python3
"""
AlphaEarth T=2 vs T=7 linear probe comparison.

Tests whether using only start+end year embeddings (T=2, 128 features)
loses signal compared to all 7 annual embeddings (T=7, 448 features).
This informs whether Part 2 can use EarlyFusion T=2 for embedding
experiments or needs Pool-7 T=7.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

REPO = Path(__file__).resolve().parents[3]
AE_DIR = REPO / "data_v2" / "AlphaEarth"
MASK_DIR = REPO / "data_v2" / "Land_take_masks_coarse"
SPLITS_CSV = REPO / "preprocessing" / "outputs" / "splits" / "part2" / "split_info.csv"
STATS_JSON = REPO / "preprocessing" / "outputs" / "normalization_stats_part2.json"

AE_N_YEARS = 7
AE_N_FEATURES = 64
# Year indices: 0=2018, 1=2019, ..., 6=2024
START_YEAR_IDX = 0  # 2018
END_YEAR_IDX = 6    # 2024


def load_normalization_stats():
    with open(STATS_JSON) as f:
        raw = json.load(f)
    return (
        np.array(raw["alphaearth"]["mean"], dtype=np.float64),
        np.array(raw["alphaearth"]["std"], dtype=np.float64),
    )


def load_tile(refid, ae_mean, ae_std):
    """Load tile, return both T=7 and T=2 feature arrays."""
    ae_path = AE_DIR / f"{refid}_VEY_Mosaic.tif"
    mask_path = MASK_DIR / f"{refid}_mask.tif"

    if not ae_path.exists() or not mask_path.exists():
        return None, None, None

    with rasterio.open(ae_path) as src:
        ae_raw = src.read().astype(np.float64)
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)

    if ae_raw.shape[0] != AE_N_YEARS * AE_N_FEATURES:
        return None, None, None

    ae = ae_raw.reshape(AE_N_YEARS, AE_N_FEATURES, ae_raw.shape[1], ae_raw.shape[2])

    h_ae, w_ae = ae.shape[2], ae.shape[3]
    h_m, w_m = mask.shape
    if h_ae != h_m or w_ae != w_m:
        h, w = min(h_ae, h_m), min(w_ae, w_m)
        ae = ae[:, :, :h, :w]
        mask = mask[:h, :w]
        ae_raw_crop = ae_raw.reshape(AE_N_YEARS, AE_N_FEATURES, h_ae, w_ae)[:, :, :h, :w]
    else:
        ae_raw_crop = ae_raw.reshape(AE_N_YEARS, AE_N_FEATURES, h_ae, w_ae)

    # Normalize
    ae = (ae - ae_mean[None, :, None, None]) / (ae_std[None, :, None, None] + 1e-8)

    # Valid pixels
    valid = np.any(ae_raw_crop != 0, axis=(0, 1))

    # T=7: all years (H*W, 448)
    feat_t7 = ae.transpose(2, 3, 0, 1).reshape(-1, AE_N_YEARS * AE_N_FEATURES)

    # T=2: start + end year only (H*W, 128)
    ae_t2 = ae[[START_YEAR_IDX, END_YEAR_IDX]]  # (2, 64, H, W)
    feat_t2 = ae_t2.transpose(2, 3, 0, 1).reshape(-1, 2 * AE_N_FEATURES)

    mask_flat = (mask > 0).astype(np.int32).ravel()
    valid_flat = valid.ravel()

    return feat_t7[valid_flat], feat_t2[valid_flat], mask_flat[valid_flat]


def subsample_balanced(features, labels, max_per_class=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n = min(len(pos_idx), len(neg_idx), max_per_class)
    pos_sel = rng.choice(pos_idx, size=n, replace=False)
    neg_sel = rng.choice(neg_idx, size=n, replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


def main():
    splits = pd.read_csv(SPLITS_CSV)
    train_refids = splits[splits["split"] == "train"]["refid"].tolist()
    test_refids = splits[splits["split"] == "test"]["refid"].tolist()

    ae_mean, ae_std = load_normalization_stats()
    rng = np.random.default_rng(42)

    # Load data
    def load_split(refids, name, max_per_tile=2000):
        all_t7, all_t2, all_labels = [], [], []
        skipped = 0
        for refid in refids:
            f7, f2, lab = load_tile(refid, ae_mean, ae_std)
            if f7 is None:
                skipped += 1
                continue
            if len(lab) > max_per_tile:
                idx = rng.choice(len(lab), size=max_per_tile, replace=False)
                f7, f2, lab = f7[idx], f2[idx], lab[idx]
            all_t7.append(f7)
            all_t2.append(f2)
            all_labels.append(lab)
        X7 = np.concatenate(all_t7)
        X2 = np.concatenate(all_t2)
        y = np.concatenate(all_labels)
        print(f"  {name}: {len(refids)} tiles ({skipped} skipped), "
              f"{len(y)} pixels, {y.sum()} pos ({100*y.mean():.1f}%)")
        return X7, X2, y

    print("Loading tiles...")
    X7_train, X2_train, y_train = load_split(train_refids, "train")
    X7_test, X2_test, y_test = load_split(test_refids, "test")

    # Balance training
    bal_idx = subsample_balanced(X7_train, y_train, max_per_class=50000, rng=rng)
    X7_bal, X2_bal, y_bal = X7_train[bal_idx], X2_train[bal_idx], y_train[bal_idx]
    print(f"\nBalanced train: {len(y_bal)} pixels ({y_bal.sum()} pos)\n")

    # Also test intermediate configs
    configs = {
        "T=2 (2018+2024)": (X2_bal, X2_test, 128),
        "T=3 (2018+2021+2024)": None,  # built below
        "T=7 (all years)": (X7_bal, X7_test, 448),
    }

    # Build T=3: start + middle + end (indices 0, 3, 6)
    mid_idx = 3  # 2021
    # Reload is wasteful; instead extract from T=7 which has all years concatenated
    # T=7 layout: [2018_feat0..63, 2019_feat0..63, ..., 2024_feat0..63]
    t3_train = np.concatenate([
        X7_bal[:, START_YEAR_IDX*64:(START_YEAR_IDX+1)*64],
        X7_bal[:, mid_idx*64:(mid_idx+1)*64],
        X7_bal[:, END_YEAR_IDX*64:(END_YEAR_IDX+1)*64],
    ], axis=1)
    t3_test = np.concatenate([
        X7_test[:, START_YEAR_IDX*64:(START_YEAR_IDX+1)*64],
        X7_test[:, mid_idx*64:(mid_idx+1)*64],
        X7_test[:, END_YEAR_IDX*64:(END_YEAR_IDX+1)*64],
    ], axis=1)
    configs["T=3 (2018+2021+2024)"] = (t3_train, t3_test, 192)

    # Also T=1 difference: end - start
    t1_diff_train = X2_bal[:, 64:] - X2_bal[:, :64]  # 2024 - 2018
    t1_diff_test = X2_test[:, 64:] - X2_test[:, :64]
    configs["T=1 diff (2024-2018)"] = (t1_diff_train, t1_diff_test, 64)

    print("=" * 70)
    print(f"{'Config':<25} {'Features':>8}  {'LR AUC':>8} {'LR F1':>7}  {'RF AUC':>8} {'RF F1':>7}")
    print("=" * 70)

    for name in ["T=1 diff (2024-2018)", "T=2 (2018+2024)", "T=3 (2018+2021+2024)", "T=7 (all years)"]:
        X_tr, X_te, n_feat = configs[name]

        # Clean any NaN from zero-padded pixel normalization
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0)

        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
        lr.fit(X_tr, y_bal)
        lr_prob = lr.predict_proba(X_te)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_prob)
        lr_f1 = f1_score(y_test, (lr_prob > 0.5).astype(int))

        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1,
                                    random_state=42, class_weight="balanced")
        rf.fit(X_tr, y_bal)
        rf_prob = rf.predict_proba(X_te)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_prob)
        rf_f1 = f1_score(y_test, (rf_prob > 0.5).astype(int))

        print(f"{name:<25} {n_feat:>8}  {lr_auc:>8.4f} {lr_f1:>7.4f}  {rf_auc:>8.4f} {rf_f1:>7.4f}")

    print("=" * 70)
    print("\nIf T=2 AUC ≈ T=7 AUC → use EarlyFusion T=2 for all Part 2 experiments")
    print("If T=7 is meaningfully better → use Pool-7 T=7 for embedding experiments")


if __name__ == "__main__":
    main()
