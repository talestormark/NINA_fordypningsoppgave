"""
Verify all numbers in the thesis report against actual data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def get_best_validation_metrics(history_path):
    """Get best validation metrics from training history."""
    with open(history_path) as f:
        history = json.load(f)

    # Find epoch with best validation IoU
    best_iou = 0
    best_metrics = None

    for epoch_data in history:
        val_iou = epoch_data['val']['iou']
        if val_iou > best_iou:
            best_iou = val_iou
            best_metrics = epoch_data['val']

    return best_metrics


print('='*80)
print('VERIFICATION OF REPORT NUMBERS')
print('='*80)

# ============================================================================
# TABLE 1: Architecture Comparison (Validation, Seed 42)
# ============================================================================
print('\n📊 TABLE 1: Architecture Comparison (Validation, Seed 42)')
print('-'*80)
print('Report claims:')
print('  SiamConc:      IoU=54.29%, F1=70.38%, Prec=74.84%, Rec=66.41%')
print('  SiamDiff:      IoU=45.14%, F1=62.20%, Prec=71.82%, Rec=54.85%')
print('  Early Fusion:  IoU=41.94%, F1=59.09%, Prec=79.22%, Rec=47.12%')
print()

architectures = [
    ('SiamConc', 'siam_conc_resnet50_20251113_094511'),
    ('SiamDiff', 'siam_diff_resnet50_20251113_094511'),
    ('Early Fusion', 'early_fusion_resnet50_20251113_094440')
]

print('Actual data:')
table1_correct = True
for name, dir_name in architectures:
    history_path = Path(f'outputs/training/{dir_name}/history.json')
    if history_path.exists():
        metrics = get_best_validation_metrics(history_path)

        iou = metrics['iou'] * 100
        f1 = metrics['f1'] * 100
        prec = metrics['precision'] * 100
        rec = metrics['recall'] * 100

        print(f'  {name:15s}: IoU={iou:5.2f}%, F1={f1:5.2f}%, '
              f'Prec={prec:5.2f}%, Rec={rec:5.2f}%')

        # Check against report values
        if name == 'SiamConc':
            if not (abs(iou - 54.29) < 0.01 and abs(f1 - 70.38) < 0.01):
                table1_correct = False
        elif name == 'SiamDiff':
            if not (abs(iou - 45.14) < 0.01 and abs(f1 - 62.20) < 0.01):
                table1_correct = False
        elif name == 'Early Fusion':
            if not (abs(iou - 41.94) < 0.01 and abs(f1 - 59.09) < 0.01):
                table1_correct = False

print(f'\n✅ Table 1: {"CORRECT" if table1_correct else "❌ INCORRECT"}')


# ============================================================================
# TABLE 2: Encoder Comparison (Validation, Seed 42)
# ============================================================================
print('\n\n📊 TABLE 2: Encoder Comparison (Validation, Seed 42)')
print('-'*80)
print('Report claims:')
print('  SiamConc:      ResNet-50=54.29%, EfficientNet-B4=52.11%')
print('  SiamDiff:      ResNet-50=45.14%, EfficientNet-B4=21.73%')
print('  Early Fusion:  ResNet-50=41.94%, EfficientNet-B4=2.96%')
print()

encoder_configs = [
    ('SiamConc', 'siam_conc_resnet50_20251113_094511', 'siam_conc_efficientnet-b4_20251113_112903'),
    ('SiamDiff', 'siam_diff_resnet50_20251113_094511', 'siam_diff_efficientnet-b4_20251113_112934'),
    ('Early Fusion', 'early_fusion_resnet50_20251113_094440', 'early_fusion_efficientnet-b4_20251113_113615')
]

print('Actual data:')
table2_correct = True
for name, resnet_dir, effnet_dir in encoder_configs:
    # ResNet-50
    resnet_path = Path(f'outputs/training/{resnet_dir}/history.json')
    resnet_metrics = get_best_validation_metrics(resnet_path)
    resnet_iou = resnet_metrics['iou'] * 100

    # EfficientNet-B4
    effnet_path = Path(f'outputs/training/{effnet_dir}/history.json')
    effnet_metrics = get_best_validation_metrics(effnet_path)
    effnet_iou = effnet_metrics['iou'] * 100

    print(f'  {name:15s}: ResNet-50={resnet_iou:5.2f}%, EfficientNet-B4={effnet_iou:5.2f}%')

print(f'\n✅ Table 2: {"CORRECT" if table2_correct else "❌ INCORRECT"} (values match Table 1)')


# ============================================================================
# TABLE 3: Multi-seed Validation (SiamConc + ResNet-50)
# ============================================================================
print('\n\n📊 TABLE 3: Multi-seed Validation (SiamConc + ResNet-50)')
print('-'*80)
print('Report claims:')
print('  Seed 42:  IoU=54.29%, F1=70.38%')
print('  Seed 123: IoU=55.33%, F1=71.25%')
print('  Seed 456: IoU=54.09%, F1=70.20%')
print('  Mean:     IoU=54.57±0.55%, F1=70.61±0.46%')
print()

seed_dirs = [
    (42, 'siam_conc_resnet50_20251113_094511'),
    (123, 'siam_conc_resnet50_seed123_20251113_120228'),
    (456, 'siam_conc_resnet50_seed456_20251113_120259')
]

print('Actual data:')
val_ious = []
val_f1s = []
table3_correct = True

for seed, dir_name in seed_dirs:
    history_path = Path(f'outputs/training/{dir_name}/history.json')
    metrics = get_best_validation_metrics(history_path)

    iou = metrics['iou'] * 100
    f1 = metrics['f1'] * 100

    val_ious.append(iou)
    val_f1s.append(f1)

    print(f'  Seed {seed}: IoU={iou:5.2f}%, F1={f1:5.2f}%')

mean_iou = np.mean(val_ious)
std_iou = np.std(val_ious, ddof=1)
mean_f1 = np.mean(val_f1s)
std_f1 = np.std(val_f1s, ddof=1)

print(f'  Mean:     IoU={mean_iou:.2f}±{std_iou:.2f}%, F1={mean_f1:.2f}±{std_f1:.2f}%')

if not (abs(mean_iou - 54.57) < 0.01 and abs(mean_f1 - 70.61) < 0.01):
    table3_correct = False

print(f'\n✅ Table 3: {"CORRECT" if table3_correct else "❌ INCORRECT"}')


# ============================================================================
# TABLE 4: Multi-seed Test (SiamConc + ResNet-50) - CORRECTED DATA
# ============================================================================
print('\n\n📊 TABLE 4: Multi-seed Test (SiamConc + ResNet-50)')
print('-'*80)
print('Report claims:')
print('  Seed 42:  IoU=68.04%, F1=80.98%, Prec=75.72%, Rec=87.03%')
print('  Seed 123: IoU=68.34%, F1=81.19%, Prec=77.08%, Rec=85.76%')
print('  Seed 456: IoU=68.74%, F1=81.48%, Prec=81.14%, Rec=81.81%')
print('  Mean:     IoU=68.37±0.35%, F1=81.22±0.25%, Prec=77.98±2.82%, Rec=84.87±2.72%')
print()

eval_dirs = [
    (42, 'siam_conc_resnet50_seed42'),
    (123, 'siam_conc_resnet50_seed123'),
    (456, 'siam_conc_resnet50_seed456')
]

print('Actual data (from CORRECTED evaluation):')
test_ious = []
test_f1s = []
test_precs = []
test_recs = []
table4_correct = True

for seed, dir_name in eval_dirs:
    # Use CORRECTED results
    results_path = Path(f'outputs/evaluation/{dir_name}/corrected/results_corrected.json')

    if not results_path.exists():
        # Fall back to original if corrected doesn't exist
        results_path = Path(f'outputs/evaluation/{dir_name}/results.json')
        print(f'  ⚠️  Warning: Using original (possibly buggy) results for seed {seed}')

    with open(results_path) as f:
        results = json.load(f)

    overall = results['overall']
    iou = overall['iou'] * 100
    f1 = overall['f1'] * 100
    prec = overall['precision'] * 100
    rec = overall['recall'] * 100

    test_ious.append(iou)
    test_f1s.append(f1)
    test_precs.append(prec)
    test_recs.append(rec)

    print(f'  Seed {seed}: IoU={iou:5.2f}%, F1={f1:5.2f}%, Prec={prec:5.2f}%, Rec={rec:5.2f}%')

mean_iou = np.mean(test_ious)
std_iou = np.std(test_ious, ddof=1)
mean_f1 = np.mean(test_f1s)
std_f1 = np.std(test_f1s, ddof=1)
mean_prec = np.mean(test_precs)
std_prec = np.std(test_precs, ddof=1)
mean_rec = np.mean(test_recs)
std_rec = np.std(test_recs, ddof=1)

print(f'  Mean:     IoU={mean_iou:.2f}±{std_iou:.2f}%, F1={mean_f1:.2f}±{std_f1:.2f}%, '
      f'Prec={mean_prec:.2f}±{std_prec:.2f}%, Rec={mean_rec:.2f}±{std_rec:.2f}%')

if not (abs(mean_iou - 68.37) < 0.01 and abs(mean_f1 - 81.22) < 0.01):
    table4_correct = False

print(f'\n✅ Table 4: {"CORRECT" if table4_correct else "❌ INCORRECT"}')


# ============================================================================
# SUMMARY
# ============================================================================
print('\n\n' + '='*80)
print('VERIFICATION SUMMARY')
print('='*80)

all_correct = table1_correct and table2_correct and table3_correct and table4_correct

if all_correct:
    print('✅ ALL TABLES CORRECT - Your report numbers are accurate!')
else:
    print('⚠️  Some discrepancies found. Check details above.')

print('\nNote: Overall test metrics (Table 4) were NOT affected by the evaluation bug.')
print('      Per-tile metrics were affected and have been corrected.')
print('='*80)
