#!/usr/bin/env python3
"""
Per-sample statistical analysis for multi-temporal land-take detection experiments.

Computes per-sample IoU from out-of-fold predictions (n=45 tiles) and performs
rigorous statistical comparisons between temporal sampling conditions.

Statistical tests:
- Wilcoxon signed-rank test (paired, nonparametric)
- Permutation test (10,000 permutations)
- Bootstrap 95% CI for mean difference
- Cliff's delta effect size

Multiple comparison correction: Holm-Bonferroni

Usage:
    python statistical_analysis_persample.py --output-dir results/

References:
- Wilcoxon signed-rank: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
- Cliff's delta: https://en.wikipedia.org/wiki/Effect_size#Cliff's_delta
- Holm-Bonferroni: https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from collections import defaultdict
import warnings

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

# Import baseline utilities
sys.path.insert(0, str(parent_dir.parent / "scripts" / "modeling"))
from train import Metrics

# Import multi-temporal modules
from models_multitemporal import create_multitemporal_model
from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from PART1_multi_temporal_experiments.scripts.experiments_v2 import (
    EXPERIMENTS_V2 as EXPERIMENTS,
    V2_OUTPUTS_DIR, V2_SENTINEL_DIR, V2_MASK_DIR, V2_ANALYSIS_DIR,
    V2_SPLITS_DIR, V2_CHANGE_LEVEL_PATH, COMPARISON_FAMILIES,
)


def compute_sample_iou(pred_logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute IoU for a single sample.

    Args:
        pred_logits: Model output logits (1, H, W) or (H, W)
        mask: Ground truth mask (H, W)
        threshold: Binary threshold for predictions

    Returns:
        float: IoU value for this sample
    """
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    pred_prob = torch.sigmoid(pred_logits)
    pred_binary = (pred_prob > threshold).float()

    pred_flat = pred_binary.view(-1)
    mask_flat = mask.view(-1)

    tp = ((pred_flat == 1) & (mask_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (mask_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (mask_flat == 1)).sum().float()

    union = tp + fp + fn
    if union == 0:
        # Both prediction and ground truth are empty -> perfect match
        return 1.0

    iou = tp / union
    return iou.item()


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    model = create_multitemporal_model(
        config.get('model_name', 'lstm_unet'),
        encoder_name=config.get('encoder_name', 'resnet50'),
        encoder_weights=None,
        in_channels=9,
        classes=1,
        lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        convlstm_kernel_size=config.get('convlstm_kernel_size', 3),
        skip_aggregation=config.get('skip_aggregation', 'max'),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both raw state dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def get_out_of_fold_predictions(experiment_config: dict, num_folds: int, device: torch.device,
                                experiments_dir: Path = None, sentinel2_dir: Path = None,
                                mask_dir: Path = None, splits_dir: Path = None,
                                change_level_path: Path = None):
    """
    Get per-sample IoU from out-of-fold predictions.

    For each fold, load the model trained on that fold and evaluate on its
    validation set (the held-out fold).

    Args:
        experiment_config: Dict with 'name' and 'temporal_sampling'
        num_folds: Number of CV folds
        device: Torch device

    Returns:
        dict: {refid: iou} for all CV samples
    """
    if experiments_dir is None:
        experiments_dir = V2_OUTPUTS_DIR
    if sentinel2_dir is None:
        sentinel2_dir = V2_SENTINEL_DIR
    if mask_dir is None:
        mask_dir = V2_MASK_DIR
    if splits_dir is None:
        splits_dir = V2_SPLITS_DIR
    if change_level_path is None:
        change_level_path = V2_CHANGE_LEVEL_PATH

    exp_name = experiment_config['name']
    temporal_sampling = experiment_config['temporal_sampling']

    per_sample_iou = {}

    for fold in range(num_folds):
        exp_dir = experiments_dir / f"{exp_name}_fold{fold}"

        if not exp_dir.exists():
            print(f"  WARNING: {exp_dir} not found, skipping fold {fold}")
            continue

        # Load config
        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Load model
        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"  WARNING: {checkpoint_path} not found, skipping fold {fold}")
            continue

        model = load_model_from_checkpoint(checkpoint_path, config, device)

        # Create dataloader for this fold's validation set
        dataloaders = get_dataloaders(
            temporal_sampling=temporal_sampling,
            batch_size=1,
            num_workers=4,
            image_size=config['image_size'],
            output_format="LSTM",
            fold=fold,
            num_folds=num_folds,
            seed=config.get('seed', 42),
            sentinel2_dir=sentinel2_dir,
            mask_dir=mask_dir,
            splits_dir=splits_dir,
            change_level_path=change_level_path,
        )
        val_loader = dataloaders['val']

        # Evaluate on validation set (out-of-fold predictions)
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                refids = batch['refid']

                outputs = model(images)

                for i, refid in enumerate(refids):
                    iou = compute_sample_iou(outputs[i], masks[i])
                    per_sample_iou[refid] = iou

    return per_sample_iou


def wilcoxon_signed_rank_test(differences: np.ndarray):
    """
    Perform Wilcoxon signed-rank test.

    Args:
        differences: Array of paired differences

    Returns:
        tuple: (statistic, p_value)
    """
    # Remove zero differences (ties)
    nonzero_diff = differences[differences != 0]

    if len(nonzero_diff) < 10:
        warnings.warn(f"Only {len(nonzero_diff)} non-zero differences. Results may be unreliable.")

    statistic, p_value = stats.wilcoxon(nonzero_diff, alternative='two-sided')

    return statistic, p_value


def permutation_test(differences: np.ndarray, n_permutations: int = 10000, seed: int = 42):
    """
    Perform permutation test on paired differences.

    Under the null hypothesis, the sign of each difference is arbitrary.
    We randomly flip signs many times and compute how extreme the observed
    mean is relative to the permutation distribution.

    Args:
        differences: Array of paired differences
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        tuple: (observed_mean, p_value)
    """
    rng = np.random.RandomState(seed)

    observed_mean = np.mean(differences)
    n = len(differences)

    # Generate permutation distribution
    permuted_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Randomly flip signs
        signs = rng.choice([-1, 1], size=n)
        permuted_diff = differences * signs
        permuted_means[i] = np.mean(permuted_diff)

    # Two-sided p-value: proportion of permuted means at least as extreme
    p_value = np.mean(np.abs(permuted_means) >= np.abs(observed_mean))

    return observed_mean, p_value


def bootstrap_ci(differences: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42):
    """
    Compute bootstrap confidence interval for mean difference.

    Args:
        differences: Array of paired differences
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level
        seed: Random seed

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(differences)

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_sample = differences[indices]
        bootstrap_means[i] = np.mean(bootstrap_sample)

    # Percentile method
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return lower, upper


def cliffs_delta(x: np.ndarray, y: np.ndarray):
    """
    Compute Cliff's delta effect size.

    Cliff's delta is the probability that a randomly selected value from x
    is greater than a randomly selected value from y, minus the probability
    that it's less.

    delta = (# pairs where x > y - # pairs where x < y) / (n_x * n_y)

    Interpretation (Romano et al., 2006):
    - |delta| < 0.147: negligible
    - 0.147 <= |delta| < 0.33: small
    - 0.33 <= |delta| < 0.474: medium
    - |delta| >= 0.474: large

    For paired data, we compare differences to zero.

    Args:
        x: First condition values
        y: Second condition values (for paired: differences, for unpaired: second group)

    Returns:
        float: Cliff's delta
    """
    # For paired differences, compare to zero
    differences = x - y
    n = len(differences)

    # Count dominance
    n_greater = np.sum(differences > 0)
    n_less = np.sum(differences < 0)

    delta = (n_greater - n_less) / n

    return delta


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def holm_bonferroni_correction(p_values: list) -> list:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values

    Returns:
        list: Adjusted p-values
    """
    n = len(p_values)

    # Sort p-values and track original indices
    indexed_pvalues = [(p, i) for i, p in enumerate(p_values)]
    indexed_pvalues.sort(key=lambda x: x[0])

    adjusted = [None] * n
    cumulative_min = 1.0

    # Process from largest to smallest (reverse order for Holm)
    for rank in range(n, 0, -1):
        p, orig_idx = indexed_pvalues[rank - 1]
        # Multiply by (n - rank + 1)
        adjusted_p = min(p * (n - rank + 1), 1.0)
        # Ensure monotonicity
        cumulative_min = min(cumulative_min, adjusted_p)
        adjusted[orig_idx] = cumulative_min

    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Per-sample statistical analysis")
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--experiments-dir', type=str, default=None,
                        help='Base directory for experiment checkpoints')
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated list of experiment keys (e.g., annual,bi_temporal,bi_seasonal)')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n-permutations', type=int, default=10000,
                        help='Number of permutations for permutation test')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap samples for CI')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Resolve paths
    experiments_dir = Path(args.experiments_dir) if args.experiments_dir else None

    # Filter experiments if requested
    if args.experiments:
        exp_keys = [k.strip() for k in args.experiments.split(',')]
        active_experiments = {k: v for k, v in EXPERIMENTS.items() if k in exp_keys}
    else:
        active_experiments = EXPERIMENTS

    print("\n" + "="*80)
    print("PER-SAMPLE STATISTICAL ANALYSIS")
    print("="*80)
    print(f"\nExperiments: {list(active_experiments.keys())}")
    print(f"Collecting out-of-fold predictions...")

    # Collect per-sample IoU for each condition
    condition_iou = {}

    for condition_name, config in active_experiments.items():
        print(f"\n--- {condition_name.upper()} (T={config['T']}) ---")
        print(f"  Experiment: {config['name']}")

        iou_dict = get_out_of_fold_predictions(
            config, args.num_folds, device,
            experiments_dir=experiments_dir,
        )
        condition_iou[condition_name] = iou_dict

        print(f"  Samples collected: {len(iou_dict)}")
        iou_values = np.array(list(iou_dict.values()))
        print(f"  Mean IoU: {np.mean(iou_values)*100:.2f}%")
        print(f"  Std IoU:  {np.std(iou_values, ddof=1)*100:.2f}%")
        print(f"  Median IoU: {np.median(iou_values)*100:.2f}%")

    # Verify all conditions have the same samples
    first_cond = next(iter(condition_iou))
    refids = set(condition_iou[first_cond].keys())
    for condition in condition_iou:
        if set(condition_iou[condition].keys()) != refids:
            raise ValueError(f"Sample mismatch between conditions! {condition} has {len(condition_iou[condition])} vs {len(refids)}")

    n_samples = len(refids)
    print(f"\n✓ All conditions have {n_samples} matching samples")

    # Create aligned arrays for paired analysis
    refid_list = sorted(refids)

    # Build arrays dynamically from available conditions
    cond_arrays = {}
    for cond_name in active_experiments:
        cond_arrays[cond_name] = np.array([condition_iou[cond_name][r] for r in refid_list])

    # Build comparisons from COMPARISON_FAMILIES, filtering to available conditions
    comparisons = []
    family_indices = {}  # family_name -> list of comparison indices

    for family_name, family_info in COMPARISON_FAMILIES.items():
        family_indices[family_name] = []
        for comp_name, cond1_key, cond2_key in family_info['comparisons']:
            if cond1_key in cond_arrays and cond2_key in cond_arrays:
                idx = len(comparisons)
                comparisons.append((comp_name, cond_arrays[cond1_key], cond_arrays[cond2_key]))
                family_indices[family_name].append(idx)

    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)

    results = []
    raw_p_values = []

    for name, cond1, cond2 in comparisons:
        print(f"\n--- {name} ---")

        # Paired differences (positive = cond1 is better)
        differences = cond1 - cond2

        # Descriptive statistics
        mean_diff = np.mean(differences)
        median_diff = np.median(differences)
        std_diff = np.std(differences, ddof=1)

        print(f"\n  Paired Differences (n={len(differences)}):")
        print(f"    Mean:   {mean_diff*100:+.2f} pp")
        print(f"    Median: {median_diff*100:+.2f} pp")
        print(f"    Std:    {std_diff*100:.2f} pp")

        # Wilcoxon signed-rank test
        w_stat, w_pvalue = wilcoxon_signed_rank_test(differences)
        print(f"\n  Wilcoxon Signed-Rank Test:")
        print(f"    W-statistic: {w_stat:.1f}")
        print(f"    p-value:     {w_pvalue:.4f}")

        # Permutation test
        perm_mean, perm_pvalue = permutation_test(
            differences, n_permutations=args.n_permutations, seed=args.seed
        )
        print(f"\n  Permutation Test ({args.n_permutations:,} permutations):")
        print(f"    Observed mean: {perm_mean*100:+.2f} pp")
        print(f"    p-value:       {perm_pvalue:.4f}")

        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_ci(
            differences, n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        print(f"\n  Bootstrap 95% CI for Mean Difference:")
        print(f"    [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}] pp")

        # Cliff's delta effect size
        delta = cliffs_delta(cond1, cond2)
        delta_interp = interpret_cliffs_delta(delta)
        print(f"\n  Effect Size:")
        print(f"    Cliff's delta: {delta:+.3f} ({delta_interp})")

        # Store results
        results.append({
            'comparison': name,
            'n': len(differences),
            'mean_diff_pp': mean_diff * 100,
            'median_diff_pp': median_diff * 100,
            'std_diff_pp': std_diff * 100,
            'ci_lower_pp': ci_lower * 100,
            'ci_upper_pp': ci_upper * 100,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'permutation_pvalue': perm_pvalue,
            'cliffs_delta': delta,
            'cliffs_delta_interpretation': delta_interp,
        })

        # Use permutation p-value for multiple comparison correction
        raw_p_values.append(perm_pvalue)

    # Apply Holm-Bonferroni correction within hypothesis families
    adjusted_p_values = [None] * len(raw_p_values)
    families = {}  # for printing

    for family_name, indices in family_indices.items():
        if not indices:
            continue
        families[family_name] = indices
        family_raw = [raw_p_values[i] for i in indices]
        family_adj = holm_bonferroni_correction(family_raw)
        for idx, adj_p in zip(indices, family_adj):
            adjusted_p_values[idx] = adj_p

    print("\n" + "="*80)
    print("MULTIPLE COMPARISON CORRECTION (Holm-Bonferroni, per-family)")
    print("="*80)

    for family_name, indices in families.items():
        print(f"\n  Family: {family_name} ({len(indices)} tests)")
        for i in indices:
            name = comparisons[i][0]
            results[i]['adjusted_pvalue'] = adjusted_p_values[i]
            results[i]['family'] = family_name
            sig = "✓ SIGNIFICANT" if adjusted_p_values[i] < 0.05 else "not significant"
            print(f"    {name}:")
            print(f"      Raw p-value:      {raw_p_values[i]:.4f}")
            print(f"      Adjusted p-value: {adjusted_p_values[i]:.4f} ({sig} at α=0.05)")

    # Summary table for LaTeX
    print("\n" + "="*80)
    print("SUMMARY TABLE (for LaTeX)")
    print("="*80)

    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Statistical comparison of temporal sampling conditions (per-sample paired analysis, $n=45$).}")
    print("\\label{tab:stats-persample}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Comparison} & \\textbf{Mean $\\Delta$} & \\textbf{95\\% CI} & \\textbf{$p$-value} & \\textbf{$p_{adj}$} & \\textbf{Cliff's $\\delta$} \\\\")
    print("\\midrule")

    for r in results:
        sig_marker = "*" if r['adjusted_pvalue'] < 0.05 else ""
        if r['adjusted_pvalue'] < 0.01:
            sig_marker = "**"
        if r['adjusted_pvalue'] < 0.001:
            sig_marker = "***"

        # Format comparison name for LaTeX
        comp_name = r['comparison'].replace('vs', 'vs.')

        print(f"{comp_name} & ${r['mean_diff_pp']:+.2f}$ pp & "
              f"$[{r['ci_lower_pp']:+.2f}, {r['ci_upper_pp']:+.2f}]$ & "
              f"${r['permutation_pvalue']:.3f}$ & "
              f"${r['adjusted_pvalue']:.3f}{sig_marker}$ & "
              f"${r['cliffs_delta']:+.2f}$ ({r['cliffs_delta_interpretation']}) \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001 (Holm-Bonferroni adjusted)")

    # Save detailed results
    output_dir = Path(args.output_dir) if args.output_dir else V2_ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample IoU values (dynamic keys)
    per_sample_data = {'refids': refid_list}
    for cond_name, arr in cond_arrays.items():
        per_sample_data[f'{cond_name}_iou'] = arr.tolist()

    per_sample_file = output_dir / "per_sample_iou.json"
    with open(per_sample_file, 'w') as f:
        json.dump(per_sample_data, f, indent=2)
    print(f"\n✓ Per-sample IoU saved to: {per_sample_file}")

    # Save statistical results (dynamic condition summaries)
    condition_summaries = {}
    for cond_name, arr in cond_arrays.items():
        condition_summaries[cond_name] = {
            'mean_iou': float(np.mean(arr)),
            'std_iou': float(np.std(arr, ddof=1)),
            'median_iou': float(np.median(arr)),
        }

    stats_file = output_dir / "statistical_analysis_persample.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'n_samples': n_samples,
            'n_permutations': args.n_permutations,
            'n_bootstrap': args.n_bootstrap,
            'seed': args.seed,
            'comparisons': results,
            'condition_summaries': condition_summaries,
        }, f, indent=2)
    print(f"✓ Statistical results saved to: {stats_file}")


if __name__ == "__main__":
    main()
