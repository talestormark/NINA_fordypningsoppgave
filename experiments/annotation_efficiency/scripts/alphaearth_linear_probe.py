#!/usr/bin/env python3
"""
AlphaEarth Linear Probe — Sanity check for RQ3e hypothesis.

Tests whether AlphaEarth embedding features separate land-take classes
at the per-pixel level. If a simple logistic regression achieves
reasonable AUC/accuracy on per-pixel classification, the hypothesis
that "good representations enable sparse-label success" has legs.

Uses raw EPSG:4326 data (co-registered AE + masks) to avoid
reprojection dependency. All 260 tiles with both AE and coarse masks.

Outputs:
  - Per-pixel AUC, F1, accuracy, precision, recall
  - t-SNE / PCA visualization of embedding space
  - Feature importance (logistic regression coefficients)
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, classification_report,
)
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
AE_DIR = REPO / "data_v2" / "AlphaEarth"
MASK_DIR = REPO / "data_v2" / "Land_take_masks_coarse"
SPLITS_CSV = REPO / "preprocessing" / "outputs" / "splits" / "part2" / "split_info.csv"
STATS_JSON = REPO / "preprocessing" / "outputs" / "normalization_stats_part2.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "linear_probe"

AE_N_YEARS = 7
AE_N_FEATURES = 64


def load_normalization_stats():
    """Load pre-computed AlphaEarth mean/std."""
    with open(STATS_JSON) as f:
        raw = json.load(f)
    return (
        np.array(raw["alphaearth"]["mean"], dtype=np.float64),
        np.array(raw["alphaearth"]["std"], dtype=np.float64),
    )


def load_tile(refid, ae_mean, ae_std):
    """Load and normalize AlphaEarth + mask for one tile.

    Returns:
        features: (N_pixels, 7*64) array of z-scored embedding features
        labels: (N_pixels,) binary array
        None, None if files missing or shapes mismatch
    """
    ae_path = AE_DIR / f"{refid}_VEY_Mosaic.tif"
    mask_path = MASK_DIR / f"{refid}_mask.tif"

    if not ae_path.exists() or not mask_path.exists():
        return None, None

    with rasterio.open(ae_path) as src:
        ae_raw = src.read().astype(np.float64)  # (448, H, W)

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)  # (H, W)

    # Reshape AE: (448, H, W) -> (7, 64, H, W)
    if ae_raw.shape[0] != AE_N_YEARS * AE_N_FEATURES:
        return None, None
    ae = ae_raw.reshape(AE_N_YEARS, AE_N_FEATURES, ae_raw.shape[1], ae_raw.shape[2])

    # Handle shape mismatch between AE and mask
    h_ae, w_ae = ae.shape[2], ae.shape[3]
    h_m, w_m = mask.shape
    if h_ae != h_m or w_ae != w_m:
        # Use the overlap region
        h, w = min(h_ae, h_m), min(w_ae, w_m)
        ae = ae[:, :, :h, :w]
        mask = mask[:h, :w]

    # Z-score normalize per feature (broadcast over T, H, W)
    ae = (ae - ae_mean[None, :, None, None]) / (ae_std[None, :, None, None] + 1e-8)

    # Identify valid pixels (not all-zero in AE, not in zero-padded corners)
    valid = np.any(ae_raw.reshape(AE_N_YEARS, AE_N_FEATURES, h_ae, w_ae)[:, :, :ae.shape[2], :ae.shape[3]] != 0, axis=(0, 1))

    # Flatten: (H, W, 7, 64) -> (H*W, 448)
    ae_flat = ae.transpose(2, 3, 0, 1).reshape(-1, AE_N_YEARS * AE_N_FEATURES)
    mask_flat = (mask > 0).astype(np.int32).ravel()
    valid_flat = valid.ravel()

    # Keep only valid pixels
    ae_flat = ae_flat[valid_flat]
    mask_flat = mask_flat[valid_flat]

    return ae_flat, mask_flat


def subsample_balanced(features, labels, max_per_class=5000, rng=None):
    """Subsample to balanced classes, capped at max_per_class each."""
    if rng is None:
        rng = np.random.default_rng(42)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n = min(len(pos_idx), len(neg_idx), max_per_class)
    pos_sel = rng.choice(pos_idx, size=n, replace=False)
    neg_sel = rng.choice(neg_idx, size=n, replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return features[idx], labels[idx]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load splits
    splits = pd.read_csv(SPLITS_CSV)
    train_refids = splits[splits["split"] == "train"]["refid"].tolist()
    val_refids = splits[splits["split"] == "val"]["refid"].tolist()
    test_refids = splits[splits["split"] == "test"]["refid"].tolist()

    print(f"Splits: {len(train_refids)} train, {len(val_refids)} val, {len(test_refids)} test")

    # Load normalization stats
    ae_mean, ae_std = load_normalization_stats()
    print(f"AE stats loaded: mean range [{ae_mean.min():.3f}, {ae_mean.max():.3f}], "
          f"std range [{ae_std.min():.3f}, {ae_std.max():.3f}]")

    # Load data
    rng = np.random.default_rng(42)

    def load_split(refids, name, max_per_tile=2000):
        all_feats, all_labels = [], []
        skipped = 0
        for refid in refids:
            feats, labels = load_tile(refid, ae_mean, ae_std)
            if feats is None:
                skipped += 1
                continue
            # Subsample per tile to keep memory manageable
            if len(labels) > max_per_tile:
                idx = rng.choice(len(labels), size=max_per_tile, replace=False)
                feats, labels = feats[idx], labels[idx]
            all_feats.append(feats)
            all_labels.append(labels)
        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)
        print(f"  {name}: {len(refids)} tiles ({skipped} skipped), "
              f"{len(X)} pixels, {y.sum()} positive ({100*y.mean():.1f}%)")
        return X, y

    print("\nLoading tiles...")
    X_train, y_train = load_split(train_refids, "train")
    X_val, y_val = load_split(val_refids, "val")
    X_test, y_test = load_split(test_refids, "test")

    # Balance training set
    X_train_bal, y_train_bal = subsample_balanced(
        X_train, y_train, max_per_class=50000, rng=rng
    )
    print(f"\nBalanced training set: {len(X_train_bal)} pixels "
          f"({y_train_bal.sum()} pos, {(1-y_train_bal).sum():.0f} neg)")

    # -----------------------------------------------------------------------
    # Model 1: Logistic Regression
    # -----------------------------------------------------------------------
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr.fit(X_train_bal, y_train_bal)

    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_prob = lr.predict_proba(X)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        print(f"  {name}: AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  "
              f"Prec={prec:.4f}  Recall={rec:.4f}")

    # -----------------------------------------------------------------------
    # Model 2: Random Forest (lightweight, 100 trees)
    # -----------------------------------------------------------------------
    print("\n=== Random Forest (100 trees) ===")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1,
                                random_state=42, class_weight="balanced")
    rf.fit(X_train_bal, y_train_bal)

    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_prob = rf.predict_proba(X)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        print(f"  {name}: AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  "
              f"Prec={prec:.4f}  Recall={rec:.4f}")

    # -----------------------------------------------------------------------
    # Feature importance (LR coefficients, averaged over timesteps)
    # -----------------------------------------------------------------------
    coefs = lr.coef_[0].reshape(AE_N_YEARS, AE_N_FEATURES)  # (7, 64)
    mean_importance = np.abs(coefs).mean(axis=0)  # (64,) avg across years
    top_features = np.argsort(mean_importance)[::-1][:10]
    print(f"\nTop 10 features by |LR coef| (averaged over years): {top_features.tolist()}")

    # -----------------------------------------------------------------------
    # Visualization: PCA of embeddings colored by label
    # -----------------------------------------------------------------------
    print("\nGenerating PCA visualization...")
    vis_sample = 10000
    vis_idx = rng.choice(len(X_test), size=min(vis_sample, len(X_test)), replace=False)
    X_vis, y_vis = X_test[vis_idx], y_test[vis_idx]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_vis)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    neg_mask = y_vis == 0
    pos_mask = y_vis == 1
    ax.scatter(X_pca[neg_mask, 0], X_pca[neg_mask, 1],
               c="steelblue", alpha=0.3, s=5, label=f"No change (n={neg_mask.sum()})")
    ax.scatter(X_pca[pos_mask, 0], X_pca[pos_mask, 1],
               c="firebrick", alpha=0.5, s=5, label=f"Land take (n={pos_mask.sum()})")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("AlphaEarth embeddings — PCA colored by land-take label")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_embeddings.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'pca_embeddings.png'}")

    # -----------------------------------------------------------------------
    # Visualization: Feature importance bar chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(range(AE_N_FEATURES), mean_importance, color="steelblue")
    ax.set_xlabel("AlphaEarth feature index")
    ax.set_ylabel("Mean |LR coefficient| across years")
    ax.set_title("Feature importance for land-take classification")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'feature_importance.png'}")

    # -----------------------------------------------------------------------
    # Visualization: Temporal coefficient heatmap
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    im = ax.imshow(np.abs(coefs), aspect="auto", cmap="viridis")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Year")
    ax.set_yticks(range(AE_N_YEARS))
    ax.set_yticklabels([str(y) for y in range(2018, 2025)])
    ax.set_title("LR coefficient magnitude by year and feature")
    plt.colorbar(im, ax=ax, label="|coefficient|")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "temporal_coefficients.png", dpi=150)
    print(f"  Saved: {OUT_DIR / 'temporal_coefficients.png'}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"Test AUC — Logistic Regression: {auc_lr:.4f}")
    print(f"Test AUC — Random Forest:       {auc_rf:.4f}")
    if auc_lr > 0.75 or auc_rf > 0.75:
        print("=> AlphaEarth features SEPARATE land-take classes well.")
        print("   RQ3e hypothesis is viable — proceed with full experiments.")
    elif auc_lr > 0.65 or auc_rf > 0.65:
        print("=> AlphaEarth features show MODERATE separation.")
        print("   RQ3e may work but expect a gap vs. dense U-Net.")
    else:
        print("=> AlphaEarth features show WEAK separation.")
        print("   RQ3e hypothesis is at risk — reconsider approach.")


if __name__ == "__main__":
    main()
