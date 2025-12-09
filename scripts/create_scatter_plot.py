"""
Create scatter plot showing relationship between land take change ratio and model performance.

For each test tile, plots:
- x-axis: fraction of pixels with land take in ground truth (change_ratio)
- y-axis: mean IoU and F1 across three model seeds

Data: siam_conc_resnet50 model evaluated on test tiles
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

# Paths to evaluation results (CORRECTED data)
base_path = Path("outputs/evaluation")
seed_dirs = [
    "siam_conc_resnet50_seed42",
    "siam_conc_resnet50_seed123",
    "siam_conc_resnet50_seed456"
]

# Load CORRECTED data from all three seeds
dfs = []
for seed_dir in seed_dirs:
    # Use corrected CSV files (fixed double-sigmoid bug)
    csv_path = base_path / seed_dir / "corrected" / "per_tile_results_corrected.csv"
    df = pd.read_csv(csv_path)
    df['seed'] = seed_dir
    dfs.append(df)

# Combine all data
all_data = pd.concat(dfs, ignore_index=True)

# Calculate mean performance per tile across seeds
tile_means = all_data.groupby('tile_id').agg({
    'change_ratio': 'mean',  # Should be same across seeds
    'iou': 'mean',
    'f1': 'mean'
}).reset_index()

# Sort by change_ratio for easier visualization
tile_means = tile_means.sort_values('change_ratio')

print("Per-tile mean performance across 3 seeds:")
print(tile_means.to_string(index=False))
print(f"\nTotal test tiles: {len(tile_means)}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Change Ratio vs IoU
ax1.scatter(tile_means['change_ratio'], tile_means['iou'],
           s=100, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1)
ax1.set_xlabel('Land Take Change Ratio (fraction of pixels)')
ax1.set_ylabel('IoU (Intersection over Union)')
ax1.set_title('Model Performance vs. Change Ratio')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(-0.02, max(tile_means['change_ratio']) * 1.1)
ax1.set_ylim(-0.05, max(tile_means['iou']) * 1.1)

# Add text annotation for correlation
ax1.text(0.05, 0.95, f'r = {np.corrcoef(tile_means["change_ratio"], tile_means["iou"])[0,1]:.3f}',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Change Ratio vs F1
ax2.scatter(tile_means['change_ratio'], tile_means['f1'],
           s=100, alpha=0.7, color='coral', edgecolors='black', linewidth=1)
ax2.set_xlabel('Land Take Change Ratio (fraction of pixels)')
ax2.set_ylabel('F1 Score')
ax2.set_title('Model Performance vs. Change Ratio')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(-0.02, max(tile_means['change_ratio']) * 1.1)
ax2.set_ylim(-0.05, max(tile_means['f1']) * 1.1)

# Add text annotation for correlation
ax2.text(0.05, 0.95, f'r = {np.corrcoef(tile_means["change_ratio"], tile_means["f1"])[0,1]:.3f}',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save figure
output_path = Path("outputs/figures/performance_vs_change_ratio_corrected.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Also save the old name for backwards compatibility
plt.savefig(Path("outputs/figures/performance_vs_change_ratio.png"), dpi=300, bbox_inches='tight')

# Calculate correlation coefficients
corr_iou = np.corrcoef(tile_means['change_ratio'], tile_means['iou'])[0, 1]
corr_f1 = np.corrcoef(tile_means['change_ratio'], tile_means['f1'])[0, 1]

print(f"\nCorrelation coefficients:")
print(f"  Change Ratio vs IoU: {corr_iou:.3f}")
print(f"  Change Ratio vs F1:  {corr_f1:.3f}")

# Statistical interpretation
print("\n" + "="*70)
print("INTERPRETATION (CORRECTED DATA)")
print("="*70)

# Check mean recall to verify we have correct data
mean_recall = all_data['recall'].mean()
print(f"\nMean recall across all tiles and seeds: {mean_recall:.3f}")

# Correlation-based interpretation
if corr_iou > 0.7:
    iou_strength = "strong positive"
elif corr_iou > 0.4:
    iou_strength = "moderate positive"
elif corr_iou > 0.1:
    iou_strength = "weak positive"
elif corr_iou > -0.1:
    iou_strength = "negligible"
elif corr_iou > -0.4:
    iou_strength = "weak negative"
elif corr_iou > -0.7:
    iou_strength = "moderate negative"
else:
    iou_strength = "strong negative"

print(f"\nThe scatter plot shows a {iou_strength} correlation (r={corr_iou:.3f}) between")
print("the fraction of land take pixels and model IoU performance.")

if corr_iou > 0.4:
    print("\nThe model performs BETTER on tiles with MORE land take change. This suggests")
    print("the model has learned to detect land take more reliably when there is more")
    print("change signal present in the tile. Possible explanations:")
    print("  • More training examples of change patterns in high-change tiles")
    print("  • Easier detection when change signal is stronger")
    print("  • Less class imbalance in high-change tiles")
elif corr_iou < -0.4:
    print("\nThe model performs WORSE on tiles with MORE land take change. This suggests")
    print("the model struggles with tiles that have extensive land take, possibly due to:")
    print("  • Increased complexity in high-change areas")
    print("  • Harder boundaries or more fragmented change patterns")
    print("  • Systematic differences in change characteristics")
else:
    print("\nThe correlation is weak, suggesting that the amount of land take in a tile")
    print("is not a strong predictor of model performance. Performance may be influenced")
    print("more by other factors such as:")
    print("  • Spatial patterns and boundary complexity")
    print("  • Specific land cover types involved in the change")
    print("  • Image quality and regional characteristics")
    print("  • Tile-specific challenges (occlusion, shadows, etc.)")

# Performance range analysis
min_change = tile_means['change_ratio'].min()
max_change = tile_means['change_ratio'].max()
min_iou = tile_means['iou'].min()
max_iou = tile_means['iou'].max()

print(f"\nPerformance range:")
print(f"  Change ratio ranges from {min_change:.1%} to {max_change:.1%}")
print(f"  IoU ranges from {min_iou:.3f} to {max_iou:.3f}")

if mean_recall < 0.999:
    print(f"  This represents a {(max_iou/min_iou - 1)*100:.1f}% relative difference in IoU")
    print(f"  across different land take levels.")

print("\nNote: Each point represents the mean performance across 3 model seeds,")
print("providing a robust estimate of typical performance for each tile.")
print("="*70)
