#!/usr/bin/env python3
"""
Per-sample statistical analysis for 1D vs 2D temporal modeling (RQ2c).

Compares per-pixel (1x1) vs patch-based (3x3) ConvLSTM kernel sizes
using out-of-fold predictions (n=45 tiles).

Statistical tests:
- Paired permutation test (10,000 permutations) - primary
- Wilcoxon signed-rank test (confirmatory)
- Bootstrap 95% CI for mean difference
- Probability of improvement P(d_i > 0)

Usage:
    python statistical_analysis_1dvs2d.py
"""

import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import warnings

# Add paths for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))

# Import multi-temporal modules
from models_multitemporal import create_multitemporal_model
from PART1_multi_temporal_experiments.scripts.data_preparation.dataset_multitemporal import get_dataloaders
from PART1_multi_temporal_experiments.config import MT_EXPERIMENTS_DIR

# Experiment configurations
EXPERIMENTS = {
    'k3x3': {
        'name': 'exp010_lstm7_no_es',
        'temporal_sampling': 'annual',
        'convlstm_kernel_size': 3,
        'description': '3×3 patch-based (baseline)',
    },
    'k1x1': {
        'name': 'exp004_v2',
        'temporal_sampling': 'annual',
        'convlstm_kernel_size': 1,
        'description': '1×1 per-pixel',
    },
}


def compute_sample_iou(pred_logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU for a single sample."""
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
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def get_out_of_fold_predictions(experiment_config: dict, num_folds: int, device: torch.device):
    """Get per-sample IoU from out-of-fold predictions."""
    exp_name = experiment_config['name']
    temporal_sampling = experiment_config['temporal_sampling']

    per_sample_iou = {}

    for fold in range(num_folds):
        exp_dir = MT_EXPERIMENTS_DIR / f"{exp_name}_fold{fold}"

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
        )
        val_loader = dataloaders['val']

        # Evaluate on validation set
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


def permutation_test(differences: np.ndarray, n_permutations: int = 10000, seed: int = 42):
    """Perform permutation test on paired differences."""
    rng = np.random.RandomState(seed)

    observed_mean = np.mean(differences)
    n = len(differences)

    permuted_means = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        permuted_diff = differences * signs
        permuted_means[i] = np.mean(permuted_diff)

    p_value = np.mean(np.abs(permuted_means) >= np.abs(observed_mean))

    return observed_mean, p_value


def bootstrap_ci(differences: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95, seed: int = 42):
    """Compute bootstrap confidence interval for mean difference."""
    rng = np.random.RandomState(seed)
    n = len(differences)

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_sample = differences[indices]
        bootstrap_means[i] = np.mean(bootstrap_sample)

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return lower, upper


def main():
    parser = argparse.ArgumentParser(description="1D vs 2D statistical analysis")
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--n-permutations', type=int, default=10000)
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("\n" + "="*80)
    print("1D vs 2D TEMPORAL MODELING STATISTICAL ANALYSIS (RQ2c)")
    print("="*80)
    print("\nComparing ConvLSTM kernel sizes:")
    print("  - 3×3 (baseline): Patch-based temporal modeling")
    print("  - 1×1 (ablation): Per-pixel temporal modeling")
    print(f"\nCollecting out-of-fold predictions for n=45 samples per condition...")

    # Collect per-sample IoU for each condition
    condition_iou = {}

    for condition_name, config in EXPERIMENTS.items():
        print(f"\n--- {config['description']} ---")
        print(f"  Experiment: {config['name']}")

        iou_dict = get_out_of_fold_predictions(config, args.num_folds, device)
        condition_iou[condition_name] = iou_dict

        print(f"  Samples collected: {len(iou_dict)}")
        iou_values = np.array(list(iou_dict.values()))
        print(f"  Mean IoU: {np.mean(iou_values)*100:.2f}%")
        print(f"  Std IoU:  {np.std(iou_values, ddof=1)*100:.2f}%")
        print(f"  Median IoU: {np.median(iou_values)*100:.2f}%")

    # Verify all conditions have the same samples
    refids = set(condition_iou['k3x3'].keys())
    if set(condition_iou['k1x1'].keys()) != refids:
        raise ValueError("Sample mismatch between conditions!")

    n_samples = len(refids)
    print(f"\n✓ Both conditions have {n_samples} matching samples")

    # Create aligned arrays
    refid_list = sorted(refids)
    k3x3_iou = np.array([condition_iou['k3x3'][r] for r in refid_list])
    k1x1_iou = np.array([condition_iou['k1x1'][r] for r in refid_list])

    # Paired analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: 3×3 vs 1×1")
    print("="*80)

    # Paired differences (positive = 3x3 is better)
    differences = k3x3_iou - k1x1_iou

    # Descriptive statistics
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    std_diff = np.std(differences, ddof=1)

    print(f"\nPaired Differences (n={len(differences)}):")
    print(f"  Mean:   {mean_diff*100:+.2f} pp")
    print(f"  Median: {median_diff*100:+.2f} pp")
    print(f"  Std:    {std_diff*100:.2f} pp")

    # Probability of improvement
    n_3x3_wins = np.sum(differences > 0)
    n_1x1_wins = np.sum(differences < 0)
    n_ties = np.sum(differences == 0)
    prob_3x3_better = n_3x3_wins / n_samples

    print(f"\nProbability of Improvement:")
    print(f"  3×3 wins: {n_3x3_wins}/{n_samples} ({prob_3x3_better*100:.1f}%)")
    print(f"  1×1 wins: {n_1x1_wins}/{n_samples} ({n_1x1_wins/n_samples*100:.1f}%)")
    print(f"  Ties:     {n_ties}/{n_samples}")

    # Permutation test
    perm_mean, perm_pvalue = permutation_test(
        differences, n_permutations=args.n_permutations, seed=args.seed
    )
    print(f"\nPaired Permutation Test ({args.n_permutations:,} permutations):")
    print(f"  Observed mean: {perm_mean*100:+.2f} pp")
    print(f"  p-value:       {perm_pvalue:.4f}")

    # Wilcoxon signed-rank test (confirmatory)
    nonzero_diff = differences[differences != 0]
    if len(nonzero_diff) >= 10:
        w_stat, w_pvalue = stats.wilcoxon(nonzero_diff, alternative='two-sided')
        print(f"\nWilcoxon Signed-Rank Test (confirmatory):")
        print(f"  W-statistic: {w_stat:.1f}")
        print(f"  p-value:     {w_pvalue:.4f}")
    else:
        w_stat, w_pvalue = None, None
        print(f"\nWilcoxon test skipped (only {len(nonzero_diff)} non-zero differences)")

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(
        differences, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    print(f"\nBootstrap 95% CI for Mean Difference:")
    print(f"  [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}] pp")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    if perm_pvalue < 0.05:
        if mean_diff > 0:
            print("\n✓ 3×3 kernel SIGNIFICANTLY better than 1×1 (p < 0.05)")
        else:
            print("\n✓ 1×1 kernel SIGNIFICANTLY better than 3×3 (p < 0.05)")
    else:
        print(f"\n○ No significant difference between 3×3 and 1×1 (p = {perm_pvalue:.3f})")
        print("  The effect of spatial context in temporal modeling is not statistically significant.")

    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    if ci_excludes_zero:
        print(f"\n  Bootstrap CI excludes zero: [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]")
    else:
        print(f"\n  Bootstrap CI includes zero: [{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]")

    # Save results
    output_dir = MT_EXPERIMENTS_DIR.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'comparison': '3×3 vs 1×1 ConvLSTM kernel',
        'n_samples': n_samples,
        'n_permutations': args.n_permutations,
        'n_bootstrap': args.n_bootstrap,
        'seed': args.seed,
        'k3x3': {
            'experiment': EXPERIMENTS['k3x3']['name'],
            'mean_iou': float(np.mean(k3x3_iou)),
            'std_iou': float(np.std(k3x3_iou, ddof=1)),
            'median_iou': float(np.median(k3x3_iou)),
        },
        'k1x1': {
            'experiment': EXPERIMENTS['k1x1']['name'],
            'mean_iou': float(np.mean(k1x1_iou)),
            'std_iou': float(np.std(k1x1_iou, ddof=1)),
            'median_iou': float(np.median(k1x1_iou)),
        },
        'paired_analysis': {
            'mean_diff_pp': mean_diff * 100,
            'median_diff_pp': median_diff * 100,
            'std_diff_pp': std_diff * 100,
            'ci_lower_pp': ci_lower * 100,
            'ci_upper_pp': ci_upper * 100,
            'permutation_pvalue': perm_pvalue,
            'wilcoxon_pvalue': w_pvalue,
            'prob_3x3_better': prob_3x3_better,
            'n_3x3_wins': int(n_3x3_wins),
            'n_1x1_wins': int(n_1x1_wins),
        },
        'per_sample': {
            'refids': refid_list,
            'k3x3_iou': k3x3_iou.tolist(),
            'k1x1_iou': k1x1_iou.tolist(),
            'differences': differences.tolist(),
        }
    }

    output_file = output_dir / "statistical_analysis_1dvs2d.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    # Print summary for Experiments.tex
    print("\n" + "="*80)
    print("SUMMARY FOR EXPERIMENTS.TEX")
    print("="*80)
    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{1D vs 2D temporal modeling comparison (per-sample paired analysis, $n={n_samples}$).}}
\\label{{tab:1dvs2d}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Kernel}} & \\textbf{{Mean IoU}} & \\textbf{{Std}} & \\textbf{{$\\Delta$ vs 3×3}} & \\textbf{{p-value}} \\\\
\\midrule
3×3 (patch-based) & {np.mean(k3x3_iou)*100:.2f}\\% & {np.std(k3x3_iou, ddof=1)*100:.2f}\\% & --- & --- \\\\
1×1 (per-pixel) & {np.mean(k1x1_iou)*100:.2f}\\% & {np.std(k1x1_iou, ddof=1)*100:.2f}\\% & {-mean_diff*100:+.2f} pp & {perm_pvalue:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

Bootstrap 95\\% CI for mean difference: $[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]$ pp
Probability of improvement: 3×3 wins on {prob_3x3_better*100:.1f}\\% of tiles
""")


if __name__ == "__main__":
    main()
