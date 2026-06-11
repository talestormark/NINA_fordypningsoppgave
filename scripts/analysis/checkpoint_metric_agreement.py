#!/usr/bin/env python3
"""
Checkpoint-metric agreement check (Methodology Section 3.6.2).

The thesis selects each fold's checkpoint by validation MICRO IoU but reports
per-tile MACRO IoU in the CV column. Section 3.6.2 claims the two metrics "tend
to rank checkpoints similarly," so the selection-vs-reporting mismatch is
unlikely to affect the conclusions. This script tests that on the re-run
EarlyFusion T=2 baseline (exp005), where train_multitemporal.py now logs both
micro ('iou') and macro ('iou_macro') validation IoU per epoch.

For each fold:
  micro_best = argmax over epochs of val micro IoU   (the selected checkpoint)
  macro_best = argmax over epochs of val macro IoU
  regret     = macro[macro_best] - macro[micro_best]  (macro IoU lost by selecting on micro, >= 0)
  spearman   = rank correlation of micro vs macro across epochs

PASS criterion: max macro regret across folds <= 0.5 pp.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_GLOB = "experiments/exp1_temporal_sampling/outputs_v3/macrocheck_exp005_fold*"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    runs = sorted(ROOT.glob(RUN_GLOB))
    if not runs:
        print(f"No runs match {RUN_GLOB} yet.")
        return

    rows = []
    for run in runs:
        hist_path = run / "history.json"
        if not hist_path.exists():
            print(f"{run.name}: history.json missing (still training?)")
            continue
        hist = json.loads(hist_path.read_text())
        # train_multitemporal.py writes {"train": [...], "val": [...]}.
        val = hist["val"] if isinstance(hist, dict) else [h["val"] for h in hist]
        if not val or "iou_macro" not in val[0]:
            print(f"{run.name}: no 'iou_macro' in history (old trainer?) -- {len(val)} epochs")
            continue
        micro = np.array([e["iou"] for e in val], float)
        macro = np.array([e["iou_macro"] for e in val], float)
        mi, ma = int(np.argmax(micro)), int(np.argmax(macro))
        regret = float(macro[ma] - macro[mi])
        rows.append((run.name, len(val), mi, ma, abs(ma - mi),
                     float(macro[mi]), float(macro[ma]), regret, spearman(micro, macro)))

    if not rows:
        print("No finished folds with 'iou_macro' yet.")
        return

    print(f"{'fold':26s} {'epochs':>6s} {'mi_best':>7s} {'ma_best':>7s} {'|dEp|':>5s} "
          f"{'macro@mi':>9s} {'macro@ma':>9s} {'regret_pp':>9s} {'spearman':>8s}")
    for name, n, mi, ma, dep, mami, mama, reg, rho in rows:
        print(f"{name:26s} {n:6d} {mi:7d} {ma:7d} {dep:5d} "
              f"{mami*100:9.2f} {mama*100:9.2f} {reg*100:9.3f} {rho:8.3f}")

    regrets = np.array([r[7] for r in rows]) * 100
    deps = np.array([r[4] for r in rows])
    print("\nSummary:")
    print(f"  folds analysed : {len(rows)}")
    print(f"  macro regret pp: mean={regrets.mean():.3f}  max={regrets.max():.3f}")
    print(f"  |epoch gap|    : mean={deps.mean():.1f}  max={deps.max()}")
    ok = regrets.max() <= 0.5
    print(f"\n  VERDICT: {'PASS' if ok else 'CHECK'} "
          f"(max macro regret {regrets.max():.3f} pp {'<=' if ok else '>'} 0.5 pp)")


if __name__ == "__main__":
    main()
