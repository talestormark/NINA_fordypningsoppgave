#!/usr/bin/env python3
"""
Stratified performance analysis for Part II experiments.

Computes per-sample IoU from out-of-fold predictions and stratifies by:
1. Change level (low / moderate / high) from split_info.csv
2. Country (countries with >= 5 tiles pooled; rest grouped as 'Other')

Usage:
    python stratified_analysis.py --experiment A3_s2_9band
    python stratified_analysis.py --experiment A3_s2_9band --compare-to D3_s2_ae_fusion
    python stratified_analysis.py --experiment A3_s2_9band --all-available
"""

import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import importlib.util

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PART1_DIR = REPO_ROOT / "PART1_multi_temporal_experiments"
PART2_DIR = REPO_ROOT / "PART2_spectral_spatial_resolution_experiments"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))
from train import Metrics

sys.path.insert(0, str(PART1_DIR / "scripts" / "modeling"))
from models_multitemporal import create_multitemporal_model

_p2_spec = importlib.util.spec_from_file_location(
    "p2_dataset", PART2_DIR / "scripts" / "data_preparation" / "dataset.py"
)
_p2_dataset = importlib.util.module_from_spec(_p2_spec)
_p2_spec.loader.exec_module(_p2_dataset)
EXPERIMENT_CONFIGS = _p2_dataset.EXPERIMENT_CONFIGS
get_dataloaders = _p2_dataset.get_dataloaders

EXPERIMENTS_DIR = PART2_DIR / "outputs" / "experiments"
SPLITS_CSV = REPO_ROOT / "outputs" / "splits" / "split_info.csv"
GEOJSON_PATH = REPO_ROOT / "land_take_bboxes_650m_v1.geojson"
OUTPUT_DIR = PART2_DIR / "outputs" / "analysis"

NUM_FOLDS = 5
MIN_COUNTRY_TILES = 5


def load_metadata() -> pd.DataFrame:
    """Load tile metadata from split_info.csv and geojson."""
    # Split info has change_level
    df_splits = pd.read_csv(SPLITS_CSV)

    # Try to load country from geojson
    country_map = {}
    if GEOJSON_PATH.exists():
        with open(GEOJSON_PATH) as f:
            geojson = json.load(f)
        for feat in geojson['features']:
            props = feat['properties']
            plotid = props.get('PLOTID', '')
            country_map[plotid] = props.get('country', 'Unknown')

    df_splits['country'] = df_splits['refid'].map(country_map).fillna('Unknown')

    return df_splits


def compute_sample_iou(pred_logits: torch.Tensor, mask: torch.Tensor,
                       threshold: float = 0.5) -> float:
    """Compute IoU for a single sample."""
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    pred_binary = (torch.sigmoid(pred_logits) > threshold).float()
    pred_flat = pred_binary.view(-1)
    mask_flat = mask.view(-1)

    tp = ((pred_flat == 1) & (mask_flat == 1)).sum().float()
    fp = ((pred_flat == 1) & (mask_flat == 0)).sum().float()
    fn = ((pred_flat == 0) & (mask_flat == 1)).sum().float()

    union = tp + fp + fn
    if union == 0:
        return 1.0
    return (tp / union).item()


def load_model(exp_dir: Path, config: dict, device: torch.device):
    """Load model from checkpoint."""
    cfg = EXPERIMENT_CONFIGS[config['experiment']]
    _, C, _, _ = cfg["expected_shape"]

    model = create_multitemporal_model(
        config['model_name'],
        encoder_name=config['encoder_name'],
        encoder_weights=None,
        in_channels=C,
        classes=1,
        lstm_hidden_dim=config.get('lstm_hidden_dim', 512),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        convlstm_kernel_size=config.get('convlstm_kernel_size', 3),
        skip_aggregation=config.get('skip_aggregation', 'max'),
    )

    checkpoint_path = exp_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def get_oof_predictions(experiment: str, device: torch.device) -> dict:
    """Get per-sample IoU from out-of-fold (validation set) predictions."""
    per_sample_iou = {}

    for fold in range(NUM_FOLDS):
        exp_dir = EXPERIMENTS_DIR / f"{experiment}_fold{fold}"
        if not exp_dir.exists():
            print(f"  WARNING: {exp_dir} not found, skipping fold {fold}")
            continue

        with open(exp_dir / "config.json") as f:
            config = json.load(f)

        model = load_model(exp_dir, config, device)

        dataloaders = get_dataloaders(
            experiment=config['experiment'],
            batch_size=1,
            num_workers=4,
            image_size=config.get('image_size', 64),
            fold=fold,
            num_folds=NUM_FOLDS,
            seed=config.get('seed', 42),
        )
        val_loader = dataloaders['val']

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


def main():
    parser = argparse.ArgumentParser(description="Stratified analysis for Part II experiments")
    parser.add_argument('--experiment', type=str, required=True,
                        help='Primary experiment to analyze')
    parser.add_argument('--compare-to', type=str, nargs='*', default=[],
                        help='Additional experiments to compare against')
    parser.add_argument('--all-available', action='store_true',
                        help='Compare against all experiments with 5 complete folds')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("\nLoading metadata...")
    metadata = load_metadata()
    refid_to_level = dict(zip(metadata['refid'], metadata['change_level']))
    refid_to_country = dict(zip(metadata['refid'], metadata['country']))

    # Determine which experiments to process
    experiments = [args.experiment] + args.compare_to

    if args.all_available:
        # Discover all experiments with 5 folds
        for d in EXPERIMENTS_DIR.iterdir():
            if not d.is_dir() or '_fold' not in d.name:
                continue
            exp_name = d.name.rsplit('_fold', 1)[0]
            if exp_name not in experiments:
                # Check if all 5 folds exist
                all_folds = all(
                    (EXPERIMENTS_DIR / f"{exp_name}_fold{f}").exists()
                    for f in range(NUM_FOLDS)
                )
                if all_folds:
                    experiments.append(exp_name)
        experiments = sorted(set(experiments))

    print(f"\nExperiments to analyze: {experiments}")

    # Get OOF predictions for each experiment
    all_predictions = {}
    for exp_name in experiments:
        print(f"\nProcessing {exp_name}...")
        predictions = get_oof_predictions(exp_name, device)
        all_predictions[exp_name] = predictions
        print(f"  Got predictions for {len(predictions)} samples")

    # Build results DataFrame
    rows = []
    primary_exp = args.experiment

    for refid in all_predictions[primary_exp].keys():
        row = {
            'refid': refid,
            'change_level': refid_to_level.get(refid, 'Unknown'),
            'country': refid_to_country.get(refid, 'Unknown'),
        }
        for exp_name in experiments:
            iou = all_predictions[exp_name].get(refid, np.nan)
            row[f'iou_{exp_name}'] = iou * 100  # Convert to percentage

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Stratify by change level ----
    print(f"\n{'='*70}")
    print("RESULTS BY CHANGE LEVEL")
    print(f"{'='*70}")

    level_summary = []
    for level in ['low', 'moderate', 'high']:
        subset = df[df['change_level'] == level]
        n = len(subset)

        row = {'Change Level': level, 'n': n}
        print(f"\n  {level} (n={n}):")

        for exp_name in experiments:
            col = f'iou_{exp_name}'
            mean_iou = subset[col].mean()
            std_iou = subset[col].std()
            row[f'{exp_name}_mean'] = mean_iou
            row[f'{exp_name}_std'] = std_iou
            print(f"    {exp_name}: {mean_iou:.1f}% +/- {std_iou:.1f}%")

        # Delta vs primary experiment for comparison experiments
        for comp_exp in experiments[1:]:
            delta = subset[f'iou_{comp_exp}'].mean() - subset[f'iou_{primary_exp}'].mean()
            row[f'delta_{comp_exp}_vs_{primary_exp}'] = delta
            print(f"    delta({comp_exp} - {primary_exp}): {delta:+.1f} pp")

        level_summary.append(row)

    # Overall
    print(f"\n  Overall (n={len(df)}):")
    for exp_name in experiments:
        col = f'iou_{exp_name}'
        print(f"    {exp_name}: {df[col].mean():.1f}% +/- {df[col].std():.1f}%")

    # ---- Stratify by country ----
    print(f"\n{'='*70}")
    print("RESULTS BY COUNTRY")
    print(f"{'='*70}")

    country_counts = df['country'].value_counts()
    large_countries = country_counts[country_counts >= MIN_COUNTRY_TILES].index.tolist()

    # Pool small countries
    df['country_group'] = df['country'].apply(
        lambda c: c if c in large_countries else 'Other'
    )

    country_summary = []
    for country in sorted(df['country_group'].unique()):
        subset = df[df['country_group'] == country]
        n = len(subset)

        row = {'Country': country, 'n': n}
        print(f"\n  {country} (n={n}):")

        for exp_name in experiments:
            col = f'iou_{exp_name}'
            mean_iou = subset[col].mean()
            row[f'{exp_name}_mean'] = mean_iou
            print(f"    {exp_name}: {mean_iou:.1f}%")

        country_summary.append(row)

    # ---- Save outputs ----
    # Detailed CSV
    csv_path = output_dir / f"stratified_{primary_exp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed CSV: {csv_path}")

    # Level summary CSV
    level_df = pd.DataFrame(level_summary)
    level_path = output_dir / f"stratified_by_level_{primary_exp}.csv"
    level_df.to_csv(level_path, index=False)
    print(f"Level summary: {level_path}")

    # Country summary CSV
    country_df = pd.DataFrame(country_summary)
    country_path = output_dir / f"stratified_by_country_{primary_exp}.csv"
    country_df.to_csv(country_path, index=False)
    print(f"Country summary: {country_path}")

    # JSON for programmatic access
    json_results = {
        'primary_experiment': primary_exp,
        'experiments': experiments,
        'by_change_level': {},
        'by_country': {},
    }

    for level in ['low', 'moderate', 'high']:
        subset = df[df['change_level'] == level]
        json_results['by_change_level'][level] = {
            'n': int(len(subset)),
        }
        for exp_name in experiments:
            col = f'iou_{exp_name}'
            json_results['by_change_level'][level][f'{exp_name}_mean'] = float(subset[col].mean())
            json_results['by_change_level'][level][f'{exp_name}_std'] = float(subset[col].std())

    for country in sorted(df['country_group'].unique()):
        subset = df[df['country_group'] == country]
        json_results['by_country'][country] = {
            'n': int(len(subset)),
        }
        for exp_name in experiments:
            col = f'iou_{exp_name}'
            json_results['by_country'][country][f'{exp_name}_mean'] = float(subset[col].mean())

    json_path = output_dir / f"stratified_{primary_exp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON: {json_path}")

    # LaTeX table
    print(f"\n{'='*70}")
    print("LATEX TABLE (by change level)")
    print(f"{'='*70}")

    exp_headers = " & ".join([f"\\textbf{{{e.replace('_', ' ')}}}" for e in experiments])
    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{IoU stratified by change level (out-of-fold, Part II).}}
\\label{{tab:stratified-level-p2}}
\\begin{{tabular}}{{lc{' c' * len(experiments)}}}
\\toprule
\\textbf{{Level}} & $n$ & {exp_headers} \\\\
\\midrule""")

    for level in ['low', 'moderate', 'high']:
        subset = df[df['change_level'] == level]
        n = len(subset)
        vals = " & ".join([
            f"{subset[f'iou_{e}'].mean():.1f}\\%" for e in experiments
        ])
        print(f"{level.capitalize()} & {n} & {vals} \\\\")

    # Overall
    vals_all = " & ".join([f"{df[f'iou_{e}'].mean():.1f}\\%" for e in experiments])
    print(f"\\midrule")
    print(f"Overall & {len(df)} & {vals_all} \\\\")
    print(f"""\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


if __name__ == "__main__":
    main()
