"""
Fix per-tile evaluation metrics by recalculating from saved predictions.

The original evaluation had a double-sigmoid bug that corrupted per-tile metrics.
This script recalculates correct metrics from the saved prediction files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def calculate_metrics(pred, target):
    """Calculate binary segmentation metrics."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    TP = np.sum(pred & target)
    FP = np.sum(pred & ~target)
    FN = np.sum(~pred & target)
    TN = np.sum(~pred & ~target)

    epsilon = 1e-7

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
    }


def fix_evaluation_results(eval_dir):
    """
    Recalculate correct metrics from saved predictions.

    Args:
        eval_dir: Path to evaluation directory (e.g., outputs/evaluation/siam_conc_resnet50_seed42)
    """
    eval_dir = Path(eval_dir)
    pred_dir = eval_dir / 'predictions'

    if not pred_dir.exists():
        print(f"❌ Predictions directory not found: {pred_dir}")
        return None

    print(f"\n{'='*70}")
    print(f"Processing: {eval_dir.name}")
    print(f"{'='*70}")

    # Get all prediction files
    pred_files = sorted(pred_dir.glob('*_pred.npy'))

    if len(pred_files) == 0:
        print(f"❌ No prediction files found in {pred_dir}")
        return None

    # Recalculate metrics for each tile
    per_tile_results = []
    all_TP, all_FP, all_FN, all_TN = 0, 0, 0, 0

    for pred_file in pred_files:
        tile_id = pred_file.stem.replace('_pred', '')
        target_file = pred_dir / f'{tile_id}_target.npy'

        if not target_file.exists():
            print(f"⚠️  Target file not found for {tile_id}, skipping...")
            continue

        # Load predictions and targets
        pred = np.load(pred_file)
        target = np.load(target_file)

        # Calculate metrics
        metrics = calculate_metrics(pred, target)
        metrics['tile_id'] = tile_id
        metrics['change_ratio'] = target.mean()

        per_tile_results.append(metrics)

        # Accumulate for overall metrics
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)
        all_TP += np.sum(pred_bool & target_bool)
        all_FP += np.sum(pred_bool & ~target_bool)
        all_FN += np.sum(~pred_bool & target_bool)
        all_TN += np.sum(~pred_bool & ~target_bool)

    # Calculate overall metrics
    epsilon = 1e-7
    overall_precision = all_TP / (all_TP + all_FP + epsilon)
    overall_recall = all_TP / (all_TP + all_FN + epsilon)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + epsilon)
    overall_iou = all_TP / (all_TP + all_FP + all_FN + epsilon)
    overall_accuracy = (all_TP + all_TN) / (all_TP + all_TN + all_FP + all_FN + epsilon)

    overall_results = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'iou': overall_iou,
        'accuracy': overall_accuracy,
    }

    # Print comparison with old results
    print(f"\n📊 CORRECTED RESULTS:")
    print(f"   Overall - F1: {overall_f1:.4f}, IoU: {overall_iou:.4f}, "
          f"Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}")
    print(f"   Tiles processed: {len(per_tile_results)}")

    # Load old results for comparison
    old_results_file = eval_dir / 'results.json'
    if old_results_file.exists():
        with open(old_results_file, 'r') as f:
            old_results = json.load(f)

        print(f"\n📉 OLD (BUGGY) RESULTS:")
        old_overall = old_results['overall']
        print(f"   Overall - F1: {old_overall['f1']:.4f}, IoU: {old_overall['iou']:.4f}, "
              f"Precision: {old_overall['precision']:.4f}, Recall: {old_overall['recall']:.4f}")

        print(f"\n⚠️  DIFFERENCE:")
        print(f"   ΔF1:        {overall_f1 - old_overall['f1']:+.4f}")
        print(f"   ΔIoU:       {overall_iou - old_overall['iou']:+.4f}")
        print(f"   ΔPrecision: {overall_precision - old_overall['precision']:+.4f}")
        print(f"   ΔRecall:    {overall_recall - old_overall['recall']:+.4f}")

        # Check if overall metrics were also affected
        if abs(overall_recall - old_overall['recall']) < 0.01:
            print(f"\n✅ Overall metrics appear CORRECT (small difference)")
        else:
            print(f"\n❌ Overall metrics were also AFFECTED by the bug")

    # Save corrected results
    corrected_dir = eval_dir / 'corrected'
    corrected_dir.mkdir(exist_ok=True)

    # Save per-tile CSV
    df = pd.DataFrame(per_tile_results)
    csv_path = corrected_dir / 'per_tile_results_corrected.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved corrected per-tile CSV: {csv_path}")

    # Save corrected JSON (convert numpy types to Python types)
    def convert_to_python_type(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_type(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    corrected_json = {
        'overall': convert_to_python_type(overall_results),
        'per_tile': convert_to_python_type(per_tile_results),
        'note': 'Corrected metrics recalculated from saved predictions (fixed double-sigmoid bug)',
    }

    json_path = corrected_dir / 'results_corrected.json'
    with open(json_path, 'w') as f:
        json.dump(corrected_json, f, indent=2)
    print(f"💾 Saved corrected JSON: {json_path}")

    return corrected_json


def main():
    """Process all evaluation directories."""
    eval_base = Path('outputs/evaluation')

    if not eval_base.exists():
        print(f"❌ Evaluation directory not found: {eval_base}")
        return

    # Find all evaluation directories for siam_conc_resnet50
    eval_dirs = sorted(eval_base.glob('siam_conc_resnet50_*'))

    if len(eval_dirs) == 0:
        print(f"❌ No evaluation directories found in {eval_base}")
        return

    print(f"\n{'='*70}")
    print(f"FIXING EVALUATION METRICS - DOUBLE SIGMOID BUG")
    print(f"{'='*70}")
    print(f"Found {len(eval_dirs)} evaluation directories to process")

    all_results = {}

    for eval_dir in eval_dirs:
        result = fix_evaluation_results(eval_dir)
        if result:
            all_results[eval_dir.name] = result

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Processed {len(all_results)} directories successfully")
    print(f"\nCorrected results saved in: outputs/evaluation/*/corrected/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
