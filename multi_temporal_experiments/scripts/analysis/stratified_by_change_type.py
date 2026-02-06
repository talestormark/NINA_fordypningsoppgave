#!/usr/bin/env python3
"""
Stratified performance analysis by land-take change type.

Computes per-sample IoU from out-of-fold predictions and stratifies by
change_type metadata from the geojson file.

Outputs:
- Per-type IoU statistics (mean, std, n)
- Per-type temporal advantage (LSTM-7 vs LSTM-2)
- Summary table for report

Usage:
    python stratified_by_change_type.py
"""

import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add paths for imports
script_dir = Path(__file__).resolve().parent
mt_experiments_dir = script_dir.parent.parent
sys.path.insert(0, str(mt_experiments_dir))
sys.path.insert(0, str(mt_experiments_dir / "scripts" / "modeling"))

from scripts.modeling.models_multitemporal import create_multitemporal_model
from scripts.data_preparation.dataset_multitemporal import get_dataloaders

# Define paths directly (avoid config import conflicts)
MT_EXPERIMENTS_DIR = mt_experiments_dir / "outputs" / "experiments"

# Paths
GEOJSON_PATH = mt_experiments_dir.parent / "land_take_bboxes_650m_v1.geojson"
OUTPUT_DIR = MT_EXPERIMENTS_DIR / "outputs" / "analysis"

# Experiments to analyze (unified protocol)
EXPERIMENTS = {
    'lstm7': {
        'name': 'exp010_lstm7_no_es',
        'temporal_sampling': 'annual',
        'T': 7,
    },
    'lstm2': {
        'name': 'exp003_v3',
        'temporal_sampling': 'bi_temporal',
        'T': 2,
    },
    'lstm14': {
        'name': 'exp002_v3',
        'temporal_sampling': 'quarterly',
        'T': 14,
    },
}


def load_geojson_metadata():
    """Load change_type metadata from geojson, filtered to Norway."""
    with open(GEOJSON_PATH) as f:
        data = json.load(f)

    metadata = {}
    for feat in data['features']:
        props = feat['properties']
        if props.get('country') == 'NOR':
            plotid = props['PLOTID']
            metadata[plotid] = {
                'change_type': props.get('change_type', 'Unknown'),
                'loss_type': props.get('r', 'Unknown'),
            }

    print(f"Loaded metadata for {len(metadata)} Norwegian tiles")
    return metadata


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

    return (tp / union).item()


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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def main():
    print("=" * 60)
    print("Stratified Analysis by Land-Take Change Type")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load metadata
    metadata = load_geojson_metadata()

    # Get OOF predictions for each experiment
    all_predictions = {}
    for exp_key, exp_config in EXPERIMENTS.items():
        print(f"\nProcessing {exp_key} ({exp_config['name']})...")
        predictions = get_out_of_fold_predictions(exp_config, num_folds=5, device=device)
        all_predictions[exp_key] = predictions
        print(f"  Got predictions for {len(predictions)} samples")

    # Build results dataframe
    rows = []
    for refid in all_predictions['lstm7'].keys():
        # Match with metadata
        meta = metadata.get(refid, {'change_type': 'Unknown', 'loss_type': 'Unknown'})

        row = {
            'refid': refid,
            'change_type': meta['change_type'],
            'loss_type': meta['loss_type'],
            'iou_lstm7': all_predictions['lstm7'].get(refid, np.nan),
            'iou_lstm2': all_predictions['lstm2'].get(refid, np.nan),
            'iou_lstm14': all_predictions['lstm14'].get(refid, np.nan),
        }
        row['delta_lstm7_vs_lstm2'] = row['iou_lstm7'] - row['iou_lstm2']
        row['delta_lstm7_vs_lstm14'] = row['iou_lstm7'] - row['iou_lstm14']
        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert to percentages
    for col in ['iou_lstm7', 'iou_lstm2', 'iou_lstm14', 'delta_lstm7_vs_lstm2', 'delta_lstm7_vs_lstm14']:
        df[col] = df[col] * 100

    print(f"\n{'=' * 60}")
    print("RESULTS BY CHANGE TYPE")
    print("=" * 60)

    # Group by change_type
    summary_rows = []
    for change_type in sorted(df['change_type'].unique()):
        subset = df[df['change_type'] == change_type]
        n = len(subset)

        row = {
            'Change Type': change_type,
            'n': n,
            'LSTM-7 IoU': f"{subset['iou_lstm7'].mean():.1f} ± {subset['iou_lstm7'].std():.1f}",
            'LSTM-2 IoU': f"{subset['iou_lstm2'].mean():.1f} ± {subset['iou_lstm2'].std():.1f}",
            'LSTM-14 IoU': f"{subset['iou_lstm14'].mean():.1f} ± {subset['iou_lstm14'].std():.1f}",
            'Δ(7-2)': f"{subset['delta_lstm7_vs_lstm2'].mean():+.1f}",
            'Δ(7-14)': f"{subset['delta_lstm7_vs_lstm14'].mean():+.1f}",
        }
        summary_rows.append(row)

        print(f"\n{change_type} (n={n}):")
        print(f"  LSTM-7:  {subset['iou_lstm7'].mean():.1f}% ± {subset['iou_lstm7'].std():.1f}%")
        print(f"  LSTM-2:  {subset['iou_lstm2'].mean():.1f}% ± {subset['iou_lstm2'].std():.1f}%")
        print(f"  LSTM-14: {subset['iou_lstm14'].mean():.1f}% ± {subset['iou_lstm14'].std():.1f}%")
        print(f"  Δ(LSTM-7 - LSTM-2):  {subset['delta_lstm7_vs_lstm2'].mean():+.1f} pp")
        print(f"  Δ(LSTM-7 - LSTM-14): {subset['delta_lstm7_vs_lstm14'].mean():+.1f} pp")

    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'=' * 60}")
    print("RESULTS BY LOSS TYPE")
    print("=" * 60)

    for loss_type in sorted(df['loss_type'].unique()):
        subset = df[df['loss_type'] == loss_type]
        n = len(subset)

        print(f"\n{loss_type} (n={n}):")
        print(f"  LSTM-7:  {subset['iou_lstm7'].mean():.1f}% ± {subset['iou_lstm7'].std():.1f}%")
        print(f"  LSTM-2:  {subset['iou_lstm2'].mean():.1f}% ± {subset['iou_lstm2'].std():.1f}%")
        print(f"  Δ(LSTM-7 - LSTM-2):  {subset['delta_lstm7_vs_lstm2'].mean():+.1f} pp")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    csv_path = OUTPUT_DIR / "stratified_by_change_type.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Save summary CSV
    summary_path = OUTPUT_DIR / "stratified_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")

    # Save JSON for programmatic access
    json_path = OUTPUT_DIR / "stratified_by_change_type.json"
    results = {
        'by_change_type': {},
        'by_loss_type': {},
    }

    for change_type in df['change_type'].unique():
        subset = df[df['change_type'] == change_type]
        results['by_change_type'][change_type] = {
            'n': int(len(subset)),
            'iou_lstm7_mean': float(subset['iou_lstm7'].mean()),
            'iou_lstm7_std': float(subset['iou_lstm7'].std()),
            'iou_lstm2_mean': float(subset['iou_lstm2'].mean()),
            'iou_lstm2_std': float(subset['iou_lstm2'].std()),
            'iou_lstm14_mean': float(subset['iou_lstm14'].mean()),
            'iou_lstm14_std': float(subset['iou_lstm14'].std()),
            'delta_7vs2_mean': float(subset['delta_lstm7_vs_lstm2'].mean()),
            'delta_7vs14_mean': float(subset['delta_lstm7_vs_lstm14'].mean()),
        }

    for loss_type in df['loss_type'].unique():
        subset = df[df['loss_type'] == loss_type]
        results['by_loss_type'][loss_type] = {
            'n': int(len(subset)),
            'iou_lstm7_mean': float(subset['iou_lstm7'].mean()),
            'iou_lstm2_mean': float(subset['iou_lstm2'].mean()),
            'delta_7vs2_mean': float(subset['delta_lstm7_vs_lstm2'].mean()),
        }

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    # Print LaTeX table for report
    print(f"\n{'=' * 60}")
    print("LATEX TABLE FOR REPORT")
    print("=" * 60)
    print("""
\\begin{table}[h]
\\centering
\\caption{Per-type IoU for LSTM-7 (out-of-fold, $n=45$ Norwegian tiles).
Types with $n < 3$ should be interpreted with caution.}
\\label{tab:per-type-iou}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Change Type} & $n$ & \\textbf{LSTM-7} & \\textbf{LSTM-2} & \\textbf{$\\Delta$(7-2)} \\\\
\\midrule""")

    for _, row in summary_df.sort_values('n', ascending=False).iterrows():
        ct = row['Change Type']
        if len(ct) > 25:
            ct = ct[:22] + "..."
        print(f"{ct} & {row['n']} & {row['LSTM-7 IoU'].split()[0]}\\% & {row['LSTM-2 IoU'].split()[0]}\\% & {row['Δ(7-2)']} pp \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}""")

    print("\nDone!")


if __name__ == "__main__":
    main()
