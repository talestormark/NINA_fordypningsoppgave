#!/usr/bin/env python3
"""
Boundary F-score (BF) metrics for land-take segmentation evaluation.

Implements BF@k: the boundary-localised precision/recall/F1 with a tolerance
buffer of k pixels. At 10m resolution:
  - BF@1 = 10m buffer (1 pixel tolerance)
  - BF@2 = 20m buffer (2 pixel tolerance)

Uses scipy.ndimage.distance_transform_edt for efficient buffer computation.

Usage:
    # Standalone on saved predictions
    python boundary_metrics.py --predictions-dir outputs/experiments/A3_s2_9band_fold0/predictions

    # Module import
    from boundary_metrics import boundary_f_score
    bf1 = boundary_f_score(pred_mask, gt_mask, tolerance=1)
"""

import argparse
import json
import numpy as np
from scipy.ndimage import distance_transform_edt
from pathlib import Path


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask.

    A pixel is on the boundary if it is positive (1) and has at least one
    4-connected neighbor that is negative (0), OR it is on the edge of the image.

    Args:
        mask: Binary mask (H, W), values in {0, 1}

    Returns:
        Binary boundary mask (H, W)
    """
    mask = mask.astype(bool)
    # Pad to handle image edges
    padded = np.pad(mask, 1, mode='constant', constant_values=False)

    # Check 4-connected neighbors
    eroded = (
        padded[1:-1, 1:-1] &
        padded[:-2, 1:-1] &   # up
        padded[2:, 1:-1] &    # down
        padded[1:-1, :-2] &   # left
        padded[1:-1, 2:]      # right
    )

    boundary = mask & ~eroded
    return boundary.astype(np.uint8)


def boundary_f_score(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: int = 1,
) -> dict:
    """
    Compute Boundary F-score (BF@k) between prediction and ground truth.

    For each boundary pixel in the prediction, check if there is a ground truth
    boundary pixel within `tolerance` pixels (Euclidean distance). Vice versa
    for recall.

    Args:
        pred: Binary prediction mask (H, W), values in {0, 1}
        gt: Binary ground truth mask (H, W), values in {0, 1}
        tolerance: Buffer distance in pixels (1 = 10m at 10m resolution)

    Returns:
        dict with keys: bf_precision, bf_recall, bf_f1, n_pred_boundary, n_gt_boundary
    """
    pred = pred.astype(bool).astype(np.uint8)
    gt = gt.astype(bool).astype(np.uint8)

    pred_boundary = extract_boundary(pred)
    gt_boundary = extract_boundary(gt)

    n_pred = pred_boundary.sum()
    n_gt = gt_boundary.sum()

    # Edge case: no boundaries
    if n_pred == 0 and n_gt == 0:
        return {
            'bf_precision': 1.0,
            'bf_recall': 1.0,
            'bf_f1': 1.0,
            'n_pred_boundary': 0,
            'n_gt_boundary': 0,
        }
    if n_pred == 0:
        return {
            'bf_precision': 1.0,
            'bf_recall': 0.0,
            'bf_f1': 0.0,
            'n_pred_boundary': 0,
            'n_gt_boundary': int(n_gt),
        }
    if n_gt == 0:
        return {
            'bf_precision': 0.0,
            'bf_recall': 1.0,
            'bf_f1': 0.0,
            'n_pred_boundary': int(n_pred),
            'n_gt_boundary': 0,
        }

    # Distance transform from boundary pixels
    # distance_transform_edt computes distance from 0-valued pixels
    # So we invert: distance from non-boundary to nearest boundary
    gt_dist = distance_transform_edt(1 - gt_boundary)
    pred_dist = distance_transform_edt(1 - pred_boundary)

    # Precision: fraction of pred boundary pixels within tolerance of gt boundary
    pred_boundary_coords = pred_boundary.astype(bool)
    bf_precision = (gt_dist[pred_boundary_coords] <= tolerance).sum() / n_pred

    # Recall: fraction of gt boundary pixels within tolerance of pred boundary
    gt_boundary_coords = gt_boundary.astype(bool)
    bf_recall = (pred_dist[gt_boundary_coords] <= tolerance).sum() / n_gt

    # F1
    if bf_precision + bf_recall == 0:
        bf_f1 = 0.0
    else:
        bf_f1 = 2 * bf_precision * bf_recall / (bf_precision + bf_recall)

    return {
        'bf_precision': float(bf_precision),
        'bf_recall': float(bf_recall),
        'bf_f1': float(bf_f1),
        'n_pred_boundary': int(n_pred),
        'n_gt_boundary': int(n_gt),
    }


def compute_bf_from_predictions(predictions_dir: Path, tolerances: list = None):
    """
    Compute BF metrics from saved prediction .npz files.

    Args:
        predictions_dir: Directory containing {refid}.npz files
        tolerances: List of tolerance values in pixels (default: [1, 2])

    Returns:
        dict mapping refid -> {bf@1: {...}, bf@2: {...}}
    """
    if tolerances is None:
        tolerances = [1, 2]

    results = {}
    npz_files = sorted(predictions_dir.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {predictions_dir}")
        return results

    for npz_path in npz_files:
        refid = npz_path.stem
        data = np.load(npz_path)
        prob = data['prob'].astype(np.float32)
        gt = data['mask'].astype(np.uint8)

        # Squeeze channel dim if present
        if prob.ndim == 3:
            prob = prob.squeeze(0)
        if gt.ndim == 3:
            gt = gt.squeeze(0)

        pred = (prob > 0.5).astype(np.uint8)

        results[refid] = {}
        for tol in tolerances:
            bf = boundary_f_score(pred, gt, tolerance=tol)
            results[refid][f'bf@{tol}'] = bf

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute boundary F-scores from saved predictions")
    parser.add_argument('--predictions-dir', type=str, required=True,
                        help='Directory containing prediction .npz files')
    parser.add_argument('--tolerances', type=int, nargs='+', default=[1, 2],
                        help='Tolerance values in pixels (default: 1 2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: predictions_dir/boundary_metrics.json)')
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    print(f"Computing boundary metrics from: {predictions_dir}")
    print(f"Tolerances: {args.tolerances} pixels")

    results = compute_bf_from_predictions(predictions_dir, tolerances=args.tolerances)

    if not results:
        print("No results computed.")
        return

    # Print results
    for tol in args.tolerances:
        key = f'bf@{tol}'
        print(f"\n{'='*60}")
        print(f"BF@{tol} Results ({tol * 10}m tolerance)")
        print('='*60)

        print(f"\n{'RefID':<50} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print('-' * 74)

        f1_values = []
        for refid in sorted(results.keys()):
            bf = results[refid][key]
            print(f"{refid:<50} {bf['bf_precision']*100:<8.2f} "
                  f"{bf['bf_recall']*100:<8.2f} {bf['bf_f1']*100:<8.2f}")
            f1_values.append(bf['bf_f1'])

        print('-' * 74)
        print(f"{'Mean':<50} {'':<8} {'':<8} {np.mean(f1_values)*100:.2f}%")
        print(f"{'Std':<50} {'':<8} {'':<8} {np.std(f1_values)*100:.2f}%")

    # Save results
    output_path = Path(args.output) if args.output else predictions_dir / "boundary_metrics.json"
    # Convert for JSON serialization
    serializable = {}
    for refid, tol_results in results.items():
        serializable[refid] = {}
        for key, bf in tol_results.items():
            serializable[refid][key] = {k: float(v) for k, v in bf.items()}

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
