#!/usr/bin/env python3
"""
Result aggregation script for Part II experiments.

Scans all outputs/experiments/*/history.json and test_results.json files.
Produces summary tables per block (A/B/C/D) and a combined results_summary.json.

Usage:
    python aggregate_results.py
    python aggregate_results.py --experiments-dir /path/to/experiments
    python aggregate_results.py --include-test  # also aggregate test results
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PART2_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"
OUTPUT_DIR = PART2_DIR / "outputs" / "analysis"

# Block assignments
BLOCK_ASSIGNMENT = {
    # Block A: Spectral ablation
    "A1_s2_rgb": "A",
    "A2_s2_rgbnir": "A",
    "A3_s2_9band": "A",
    "A4_s2_indices": "A",
    "A5_indices_only": "A",
    "A6_temporal_diff": "A",
    # Block B: Multi-resolution encoding
    "B1_s2_stacked": "B",
    "B2_s2_bandgroup": "B",
    # Block C: PlanetScope
    "C1_s2_rgb": "C",
    "C2_ps_rgb": "C",
    "C2hm_ps_rgb_hm": "C",
    "C3_s2_ps_fusion": "C",
    # Block D: AlphaEarth
    "D1_s2_best": "D",
    "D2_alphaearth": "D",
    "D3_s2_ae_fusion": "D",
    # Sanity checks
    "LSTM7lite_sanity": "sanity",
}

# Anchor for each block (for delta computation)
BLOCK_ANCHORS = {
    "A": "A3_s2_9band",
    "B": "A3_s2_9band",   # B1 = A3 stacked baseline
    "C": "A1_s2_rgb",     # C1 = S2 RGB
    "D": "A3_s2_9band",   # D1 = best S2
}

NUM_FOLDS = 5
METRICS = ['iou', 'f1', 'precision', 'recall']


def discover_experiments(experiments_dir: Path) -> dict:
    """Discover all experiments and their available folds."""
    experiments = defaultdict(list)

    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # Parse experiment_foldN pattern
        if '_fold' in name:
            parts = name.rsplit('_fold', 1)
            exp_name = parts[0]
            try:
                fold = int(parts[1])
                experiments[exp_name].append(fold)
            except ValueError:
                continue

    # Sort folds
    for exp_name in experiments:
        experiments[exp_name].sort()

    return dict(experiments)


def load_val_metrics(experiments_dir: Path, exp_name: str, folds: list) -> dict:
    """Load best validation metrics from history.json for each fold."""
    fold_metrics = {}

    for fold in folds:
        hist_path = experiments_dir / f"{exp_name}_fold{fold}" / "history.json"
        if not hist_path.exists():
            continue

        with open(hist_path) as f:
            history = json.load(f)

        # Find best validation epoch (by IoU)
        val_epochs = history.get('val', [])
        if not val_epochs:
            continue

        best_epoch = max(range(len(val_epochs)), key=lambda i: val_epochs[i].get('iou', 0))
        best_metrics = val_epochs[best_epoch]

        fold_metrics[fold] = {
            'epoch': best_epoch + 1,
            **{m: best_metrics.get(m, 0.0) for m in METRICS},
        }

    return fold_metrics


def load_test_metrics(experiments_dir: Path, exp_name: str) -> dict:
    """Load test results from test_results.json if available."""
    test_path = experiments_dir / exp_name / "test_results.json"
    if not test_path.exists():
        return {}

    with open(test_path) as f:
        return json.load(f)


def load_boundary_metrics(experiments_dir: Path, exp_name: str, folds: list) -> dict:
    """Load boundary metrics from prediction directories."""
    fold_bf = {}

    for fold in folds:
        bf_path = experiments_dir / f"{exp_name}_fold{fold}" / "predictions" / "boundary_metrics.json"
        if not bf_path.exists():
            continue

        with open(bf_path) as f:
            bf_data = json.load(f)

        # Average across tiles
        bf1_values = []
        bf2_values = []
        for refid, tol_results in bf_data.items():
            if 'bf@1' in tol_results:
                bf1_values.append(tol_results['bf@1']['bf_f1'])
            if 'bf@2' in tol_results:
                bf2_values.append(tol_results['bf@2']['bf_f1'])

        fold_bf[fold] = {
            'bf1_mean': np.mean(bf1_values) if bf1_values else None,
            'bf2_mean': np.mean(bf2_values) if bf2_values else None,
        }

    return fold_bf


def compute_summary(fold_metrics: dict) -> dict:
    """Compute mean +/- std across folds."""
    if not fold_metrics:
        return {}

    summary = {}
    for metric in METRICS:
        values = [fold_metrics[f][metric] for f in sorted(fold_metrics.keys())
                  if metric in fold_metrics[f]]
        if values:
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            summary[f'{metric}_values'] = [float(v) for v in values]

    return summary


def format_metric(mean, std, as_pct=True):
    """Format mean +/- std string."""
    if as_pct:
        return f"{mean*100:.2f} +/- {std*100:.2f}"
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate Part II experiment results")
    parser.add_argument('--experiments-dir', type=str, default=str(EXPERIMENTS_DIR))
    parser.add_argument('--include-test', action='store_true',
                        help='Include test set results in aggregation')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("Part II Results Aggregation")
    print(f"{'='*70}")
    print(f"Experiments dir: {experiments_dir}")

    # Discover experiments
    discovered = discover_experiments(experiments_dir)
    print(f"\nDiscovered {len(discovered)} experiments:")
    for exp_name, folds in sorted(discovered.items()):
        block = BLOCK_ASSIGNMENT.get(exp_name, "?")
        print(f"  [{block}] {exp_name}: folds {folds}")

    # Load all metrics
    all_summaries = {}

    for exp_name, folds in sorted(discovered.items()):
        # Validation metrics
        val_metrics = load_val_metrics(experiments_dir, exp_name, folds)
        val_summary = compute_summary(val_metrics)

        # Boundary metrics
        bf_metrics = load_boundary_metrics(experiments_dir, exp_name, folds)

        # Test metrics
        test_data = {}
        if args.include_test:
            test_data = load_test_metrics(experiments_dir, exp_name)

        all_summaries[exp_name] = {
            'block': BLOCK_ASSIGNMENT.get(exp_name, "?"),
            'n_folds': len(val_metrics),
            'val': val_summary,
            'val_per_fold': {str(f): m for f, m in val_metrics.items()},
            'boundary': {str(f): m for f, m in bf_metrics.items()} if bf_metrics else {},
            'test': test_data,
        }

    # Print block summaries
    for block_letter in ["A", "B", "C", "D", "sanity"]:
        block_exps = {k: v for k, v in all_summaries.items()
                      if v['block'] == block_letter}
        if not block_exps:
            continue

        anchor = BLOCK_ANCHORS.get(block_letter)

        print(f"\n{'='*70}")
        print(f"BLOCK {block_letter}")
        print(f"{'='*70}")

        # Header
        header = f"{'Experiment':<25} {'Folds':<7} {'Val IoU (%)':<20} {'Val F1 (%)':<20}"
        if anchor and anchor in all_summaries:
            header += f"  {'dIoU vs anchor':<15}"
        print(header)
        print('-' * len(header))

        for exp_name in sorted(block_exps.keys()):
            s = block_exps[exp_name]
            vs = s['val']

            iou_str = format_metric(vs.get('iou_mean', 0), vs.get('iou_std', 0)) if vs else "N/A"
            f1_str = format_metric(vs.get('f1_mean', 0), vs.get('f1_std', 0)) if vs else "N/A"

            line = f"{exp_name:<25} {s['n_folds']:<7} {iou_str:<20} {f1_str:<20}"

            # Delta vs anchor
            if anchor and anchor in all_summaries and anchor != exp_name:
                anchor_iou = all_summaries[anchor]['val'].get('iou_mean', 0)
                exp_iou = vs.get('iou_mean', 0)
                delta = (exp_iou - anchor_iou) * 100
                line += f"  {delta:+.2f} pp"
            elif exp_name == anchor:
                line += f"  (anchor)"

            print(line)

    # Save combined summary
    results_summary = {
        'experiments': {},
        'blocks': {},
    }

    for exp_name, s in all_summaries.items():
        results_summary['experiments'][exp_name] = {
            'block': s['block'],
            'n_folds': s['n_folds'],
            'val_iou_mean': s['val'].get('iou_mean'),
            'val_iou_std': s['val'].get('iou_std'),
            'val_f1_mean': s['val'].get('f1_mean'),
            'val_f1_std': s['val'].get('f1_std'),
            'val_precision_mean': s['val'].get('precision_mean'),
            'val_recall_mean': s['val'].get('recall_mean'),
            'per_fold_iou': s['val'].get('iou_values', []),
        }
        if s.get('test'):
            results_summary['experiments'][exp_name]['test'] = {
                'mean_iou': s['test'].get('mean_iou'),
                'std_iou': s['test'].get('std_iou'),
            }

    # Block-level aggregation
    for block_letter in ["A", "B", "C", "D"]:
        block_exps = {k: v for k, v in results_summary['experiments'].items()
                      if v['block'] == block_letter}
        if block_exps:
            results_summary['blocks'][block_letter] = {
                'experiments': list(block_exps.keys()),
                'anchor': BLOCK_ANCHORS.get(block_letter),
            }

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n\nResults summary saved to: {summary_path}")

    # Save CSV for easy viewing
    rows = []
    for exp_name, s in sorted(all_summaries.items()):
        vs = s['val']
        rows.append({
            'experiment': exp_name,
            'block': s['block'],
            'n_folds': s['n_folds'],
            'val_iou_mean': vs.get('iou_mean'),
            'val_iou_std': vs.get('iou_std'),
            'val_f1_mean': vs.get('f1_mean'),
            'val_f1_std': vs.get('f1_std'),
            'val_precision_mean': vs.get('precision_mean'),
            'val_recall_mean': vs.get('recall_mean'),
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
