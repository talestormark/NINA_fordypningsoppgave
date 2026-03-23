#!/usr/bin/env python3
"""
Step 0: Post-training health check for Part 1 v2 experiments.

Run BEFORE launching the analysis pipeline to verify training went as expected.
No GPU required — reads only history.json and checkpoint file existence.

Usage:
    python health_check.py
    python health_check.py --experiments annual,bi_temporal,bi_seasonal
"""

import argparse
import json
import math
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent.parent  # PART1_multi_temporal_experiments/
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir.parent))  # NINA_fordypningsoppgave/

from PART1_multi_temporal_experiments.scripts.experiments_v2 import (
    EXPERIMENTS_V2, DISPLAY_NAMES,
    V2_OUTPUTS_DIR, get_experiment_dir, get_history_path,
)

NUM_FOLDS = 5

# Thresholds
CATASTROPHIC_IOU = 0.15        # Single fold below this = FAIL
UNDERFIT_IOU = 0.30            # Experiment mean best IoU below this = WARN
HIGH_CV_THRESHOLD = 0.20       # Coefficient of variation above this = WARN
OVERSHOOT_WARN = 0.10          # best_iou - final_iou > this = WARN
OVERFIT_GAP_WARN = 0.15        # train_iou - val_iou at best epoch > this = WARN
OVERFIT_GAP_FAIL = 0.25        # train_iou - val_iou at best epoch > this = FAIL
TAIL_IMPROVING_THRESH = 0.005  # Mean IoU still rising in last 40 epochs = WARN
LATE_BEST_EPOCH_FRAC = 0.95    # Best epoch in last 5% of training = WARN
VAL_LOSS_DIVERGE_THRESH = 0.05 # Val loss increase in last 40ep while train decreases = WARN


def load_history(exp_key, fold):
    """Load history.json, return None if missing."""
    path = get_history_path(exp_key, fold)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_completeness(experiments, outputs_dir):
    """Check 1: All expected files exist."""
    print("\n" + "=" * 70)
    print("CHECK 1: CHECKPOINT COMPLETENESS")
    print("=" * 70)

    status_map = {}  # (exp_key, fold) -> status
    files_to_check = ["best_model.pth", "config.json", "history.json", "final_model.pth"]

    # Header
    print(f"\n{'Experiment':<22} " + " ".join(f"{'Fold '+str(f):^12}" for f in range(NUM_FOLDS)))
    print("-" * 85)

    n_complete = 0
    n_running = 0
    n_missing = 0
    issues = []

    for exp_key, cfg in experiments.items():
        display = DISPLAY_NAMES.get(exp_key, exp_key)
        row = f"{display:<22} "
        for fold in range(NUM_FOLDS):
            exp_dir = get_experiment_dir(exp_key, fold)
            existing = [f for f in files_to_check if (exp_dir / f).exists()]

            if len(existing) == 4:
                status = "COMPLETE"
                symbol = "OK"
                n_complete += 1
            elif "history.json" in existing:
                status = "RUNNING"
                symbol = "RUNNING"
                n_running += 1
                issues.append(f"  {display} fold {fold}: no final_model.pth (still running or crashed)")
            else:
                status = "MISSING"
                symbol = "MISSING"
                n_missing += 1
                issues.append(f"  {display} fold {fold}: MISSING (no history.json)")

            status_map[(exp_key, fold)] = status
            row += f"{symbol:^12} "
        print(row)

    total = len(experiments) * NUM_FOLDS
    print(f"\nSummary: {n_complete}/{total} complete, {n_running} running, {n_missing} missing")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(issue)

    if n_missing > 0:
        verdict = "FAIL"
    elif n_running > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    print(f"\nVerdict: {verdict}")
    return verdict, status_map


def check_stability(experiments):
    """Check 2: No NaN/Inf in losses, reasonable final values."""
    print("\n" + "=" * 70)
    print("CHECK 2: TRAINING STABILITY")
    print("=" * 70)

    issues = []
    all_ok = True

    for exp_key, cfg in experiments.items():
        display = DISPLAY_NAMES.get(exp_key, exp_key)
        for fold in range(NUM_FOLDS):
            h = load_history(exp_key, fold)
            if h is None:
                continue

            train_losses = [e['loss'] for e in h['train']]
            val_losses = [e['loss'] for e in h['val']]
            val_ious = [e['iou'] for e in h['val']]

            # Check for NaN/Inf
            nan_train = any(math.isnan(l) or math.isinf(l) for l in train_losses)
            nan_val = any(math.isnan(l) or math.isinf(l) for l in val_losses)
            nan_iou = any(math.isnan(i) or math.isinf(i) for i in val_ious)

            if nan_train or nan_val or nan_iou:
                # Check if it's only in early epochs
                late_nan = any(
                    math.isnan(l) or math.isinf(l)
                    for l in train_losses[-50:] + val_losses[-50:]
                )
                severity = "FAIL" if late_nan else "WARN (early epochs only)"
                issues.append(f"  {display} fold {fold}: NaN/Inf detected — {severity}")
                if late_nan:
                    all_ok = False

            # Final loss sanity
            if val_losses and val_losses[-1] > 10.0:
                issues.append(f"  {display} fold {fold}: final val loss = {val_losses[-1]:.2f} (unusually high)")
                all_ok = False

            # Negative IoU
            if any(i < 0 for i in val_ious):
                issues.append(f"  {display} fold {fold}: negative val IoU detected")
                all_ok = False

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(issue)
    else:
        print("\n  All losses and IoUs are finite and reasonable.")

    verdict = "PASS" if all_ok and not issues else ("FAIL" if not all_ok else "WARN")
    print(f"\nVerdict: {verdict}")
    return verdict


def check_convergence(experiments):
    """Check 3: Did models converge? Best epoch, overshoot, tail trend."""
    print("\n" + "=" * 70)
    print("CHECK 3: CONVERGENCE ANALYSIS")
    print("=" * 70)

    import numpy as np

    header = (f"{'Experiment':<22} {'Best Epoch':>12} {'Best IoU':>10} "
              f"{'Final IoU':>10} {'Overshoot':>10} {'Improving?':>12}")
    print(f"\n{header}")
    print("-" * 80)

    issues = []

    for exp_key, cfg in experiments.items():
        display = DISPLAY_NAMES.get(exp_key, exp_key)

        best_epochs = []
        best_ious = []
        final_ious = []
        total_epochs_list = []
        tail_improvements = []

        for fold in range(NUM_FOLDS):
            h = load_history(exp_key, fold)
            if h is None:
                continue

            val_ious = [e['iou'] for e in h['val']]
            total_epochs = len(val_ious)
            best_iou = max(val_ious)
            best_epoch = val_ious.index(best_iou) + 1
            final_iou = val_ious[-1]

            best_epochs.append(best_epoch)
            best_ious.append(best_iou)
            final_ious.append(final_iou)
            total_epochs_list.append(total_epochs)

            # Tail improvement: mean of last 20 - mean of epochs -40 to -20
            if total_epochs >= 40:
                tail_late = sum(val_ious[-20:]) / 20
                tail_early = sum(val_ious[-40:-20]) / 20
                tail_improvements.append(tail_late - tail_early)

        if not best_epochs:
            print(f"{display:<22} {'N/A':>12}")
            continue

        mean_best_epoch = np.mean(best_epochs)
        std_best_epoch = np.std(best_epochs, ddof=1) if len(best_epochs) > 1 else 0
        mean_best_iou = np.mean(best_ious)
        mean_final_iou = np.mean(final_ious)
        overshoot = mean_best_iou - mean_final_iou
        mean_tail = np.mean(tail_improvements) if tail_improvements else 0

        improving = "Yes" if mean_tail > TAIL_IMPROVING_THRESH else "No"

        print(f"{display:<22} {mean_best_epoch:>7.0f}±{std_best_epoch:<4.0f}"
              f"{mean_best_iou:>10.4f} {mean_final_iou:>10.4f} "
              f"{overshoot:>10.4f} {improving:>12}")

        # Underfitting: mean best IoU too low
        mean_total = np.mean(total_epochs_list)
        if mean_best_iou < UNDERFIT_IOU:
            issues.append(f"  {display}: possible underfitting — mean best IoU = {mean_best_iou:.3f} (< {UNDERFIT_IOU})")

        if mean_best_epoch > mean_total * LATE_BEST_EPOCH_FRAC:
            issues.append(f"  {display}: best epoch very late ({mean_best_epoch:.0f}/{mean_total:.0f})")

        if overshoot > OVERSHOOT_WARN:
            issues.append(f"  {display}: significant overshoot ({overshoot:.3f}) — model degrades after best epoch")

        if mean_tail > TAIL_IMPROVING_THRESH:
            issues.append(f"  {display}: val IoU still improving in last 40 epochs (Δ={mean_tail:.4f})")

    if issues:
        print("\nNotes:")
        for issue in issues:
            print(issue)
    else:
        print("\n  All models converged within training budget.")

    verdict = "WARN" if issues else "PASS"
    print(f"\nVerdict: {verdict}")
    return verdict


def check_overfit_underfit(experiments):
    """Check 4: Train-val gap and val loss trajectory (overfitting/underfitting)."""
    print("\n" + "=" * 70)
    print("CHECK 4: OVERFIT / UNDERFIT ANALYSIS")
    print("=" * 70)

    import numpy as np

    header = (f"{'Experiment':<22} {'Train IoU':>10} {'Val IoU':>10} "
              f"{'Gap':>8} {'Val Loss Trend':>15}")
    print(f"\n{header}")
    print("-" * 70)

    issues = []
    has_fail = False

    for exp_key, cfg in experiments.items():
        display = DISPLAY_NAMES.get(exp_key, exp_key)

        train_ious_at_best = []
        val_ious_at_best = []
        val_loss_diverging = []

        for fold in range(NUM_FOLDS):
            h = load_history(exp_key, fold)
            if h is None:
                continue

            val_ious = [e['iou'] for e in h['val']]
            train_ious = [e['iou'] for e in h['train']]
            val_losses = [e['loss'] for e in h['val']]
            train_losses = [e['loss'] for e in h['train']]

            best_epoch_idx = val_ious.index(max(val_ious))

            train_ious_at_best.append(train_ious[best_epoch_idx])
            val_ious_at_best.append(val_ious[best_epoch_idx])

            # Val loss divergence: val loss increasing while train loss decreasing
            # in the last 40 epochs
            n = len(val_losses)
            if n >= 40:
                val_late = np.mean(val_losses[-20:])
                val_early = np.mean(val_losses[-40:-20])
                train_late = np.mean(train_losses[-20:])
                train_early = np.mean(train_losses[-40:-20])

                val_increasing = (val_late - val_early) > VAL_LOSS_DIVERGE_THRESH
                train_decreasing = (train_early - train_late) > 0
                val_loss_diverging.append(val_increasing and train_decreasing)
            else:
                val_loss_diverging.append(False)

        if not train_ious_at_best:
            print(f"{display:<22} {'N/A':>10}")
            continue

        mean_train = np.mean(train_ious_at_best)
        mean_val = np.mean(val_ious_at_best)
        gap = mean_train - mean_val
        n_diverging = sum(val_loss_diverging)
        diverge_str = f"{n_diverging}/{len(val_loss_diverging)} folds" if n_diverging > 0 else "stable"

        print(f"{display:<22} {mean_train:>10.4f} {mean_val:>10.4f} "
              f"{gap:>8.4f} {diverge_str:>15}")

        # Flag issues
        if gap > OVERFIT_GAP_FAIL:
            issues.append(f"  {display}: SEVERE overfitting — train-val gap = {gap:.3f} (> {OVERFIT_GAP_FAIL})")
            has_fail = True
        elif gap > OVERFIT_GAP_WARN:
            issues.append(f"  {display}: overfitting — train-val gap = {gap:.3f} (> {OVERFIT_GAP_WARN})")

        if n_diverging >= 3:
            issues.append(f"  {display}: val loss diverging in {n_diverging}/{len(val_loss_diverging)} folds (train↓ val↑)")

    print(f"\n  Gap = train IoU − val IoU at best epoch (higher = more overfitting)")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(issue)
    else:
        print("\n  Train-val gaps and loss trajectories look healthy.")

    verdict = "FAIL" if has_fail else ("WARN" if issues else "PASS")
    print(f"\nVerdict: {verdict}")
    return verdict


def check_fold_variance(experiments):
    """Check 5: Flag outlier folds or high variance."""
    print("\n" + "=" * 70)
    print("CHECK 5: FOLD VARIANCE")
    print("=" * 70)

    import numpy as np

    header = (f"{'Experiment':<22} " +
              " ".join(f"{'F'+str(f):>7}" for f in range(NUM_FOLDS)) +
              f" {'Mean':>8} {'Std':>7} {'CV':>6}")
    print(f"\n{header}")
    print("-" * 85)

    issues = []
    has_fail = False

    for exp_key, cfg in experiments.items():
        display = DISPLAY_NAMES.get(exp_key, exp_key)

        fold_ious = []
        for fold in range(NUM_FOLDS):
            h = load_history(exp_key, fold)
            if h is None:
                fold_ious.append(None)
                continue
            val_ious = [e['iou'] for e in h['val']]
            fold_ious.append(max(val_ious))

        valid_ious = [x for x in fold_ious if x is not None]
        if not valid_ious:
            print(f"{display:<22} {'N/A':>7}" * NUM_FOLDS)
            continue

        mean_iou = np.mean(valid_ious)
        std_iou = np.std(valid_ious, ddof=1) if len(valid_ious) > 1 else 0
        cv = (std_iou / mean_iou * 100) if mean_iou > 0 else 0

        row = f"{display:<22} "
        for iou in fold_ious:
            if iou is None:
                row += f"{'N/A':>7} "
            else:
                # Flag outliers
                marker = ""
                if std_iou > 0.01 and abs(iou - mean_iou) > 2 * std_iou:
                    marker = "*"
                if iou < CATASTROPHIC_IOU:
                    marker = "!!"
                    has_fail = True
                    issues.append(f"  {display} fold: IoU={iou:.4f} < {CATASTROPHIC_IOU} (catastrophic)")
                row += f"{iou*100:>6.1f}{marker} "

        row += f"{mean_iou*100:>7.1f}% {std_iou*100:>6.1f}  {cv:>5.1f}%"
        print(row)

        if cv > HIGH_CV_THRESHOLD * 100:
            issues.append(f"  {display}: high fold variance (CV={cv:.1f}%)")

    print("\n  * = outlier (>2σ from mean), !! = catastrophic (<15% IoU)")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(issue)
    else:
        print("\n  All folds within expected variance.")

    verdict = "FAIL" if has_fail else ("WARN" if issues else "PASS")
    print(f"\nVerdict: {verdict}")
    return verdict


def print_nan_sample_info():
    """Check 3 (informational): Explain the NaN sample warning."""
    print("\n" + "=" * 70)
    print("NOTE: NaN SAMPLE WARNING")
    print("=" * 70)
    print("""
  During training you may see warnings like:
    WARNING: NaN detected in sample <refid> after normalization. Replacing with 0.

  This is expected and benign:
  - Source: dataset_multitemporal.py z-score normalization
  - Cause: some S2 tiles have nodata pixels (value 0). After z-score
    normalization, bands with zero variance in nodata regions produce NaN.
  - Fix: the code replaces NaN with 0.0 (neutral in normalized space)
  - Impact: none, as long as Check 2 (stability) passes — losses and IoU
    remain finite and reasonable.
""")


def main():
    parser = argparse.ArgumentParser(description="Post-training health check")
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated experiment keys (default: all)')
    parser.add_argument('--outputs-dir', type=str, default=None,
                        help='Override outputs directory')
    args = parser.parse_args()

    if args.experiments:
        exp_keys = [k.strip() for k in args.experiments.split(',')]
        experiments = {k: v for k, v in EXPERIMENTS_V2.items() if k in exp_keys}
    else:
        experiments = EXPERIMENTS_V2

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else V2_OUTPUTS_DIR

    print("=" * 70)
    print("POST-TRAINING HEALTH CHECK — Part 1 v2")
    print("=" * 70)
    print(f"Outputs dir: {outputs_dir}")
    print(f"Experiments: {list(experiments.keys())}")

    # Run all checks
    v1, status_map = check_completeness(experiments, outputs_dir)
    v2 = check_stability(experiments)
    print_nan_sample_info()
    v3 = check_convergence(experiments)
    v4 = check_overfit_underfit(experiments)
    v5 = check_fold_variance(experiments)

    # Final summary
    verdicts = {
        'Completeness': v1,
        'Stability': v2,
        'Convergence': v3,
        'Overfit/Underfit': v4,
        'Fold Variance': v5,
    }

    print("\n" + "=" * 70)
    print("HEALTH CHECK SUMMARY")
    print("=" * 70)
    for name, v in verdicts.items():
        print(f"  {name:<20} {v}")

    if any(v == "FAIL" for v in verdicts.values()):
        overall = "NO-GO — fix issues before running analysis"
    elif any(v == "WARN" for v in verdicts.values()):
        overall = "CAUTION — proceed but investigate warnings"
    else:
        overall = "GO — safe to launch analysis pipeline"

    print(f"\n  >>> {overall} <<<\n")


if __name__ == "__main__":
    main()
