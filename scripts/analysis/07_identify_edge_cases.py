"""
Script to identify edge cases based on change ratio categories
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import REPORTS_DIR


def categorize_tiles(df):
    """
    Categorize tiles based on change ratio

    Returns:
        dict: {category: list_of_refids}
    """
    categories = {
        'zero_change': [],
        'low_change': [],
        'moderate_change': [],
        'high_change': []
    }

    for idx, row in df.iterrows():
        refid = row['refid']
        change_ratio = row['change_ratio']

        if pd.isna(change_ratio):
            continue

        if change_ratio == 0:
            categories['zero_change'].append(refid)
        elif 0 < change_ratio < 5:
            categories['low_change'].append(refid)
        elif 5 <= change_ratio < 30:
            categories['moderate_change'].append(refid)
        else:  # change_ratio >= 30
            categories['high_change'].append(refid)

    return categories


if __name__ == "__main__":
    try:
        # Load mask analysis results
        analysis_file = Path(REPORTS_DIR) / "mask_analysis.csv"
        df = pd.read_csv(analysis_file)

        print("\nCategorizing tiles by change ratio...\n")
        print("=" * 80)

        # Categorize tiles
        categories = categorize_tiles(df)

        # Print summary for each category
        category_labels = {
            'zero_change': 'Zero Change (0%)',
            'low_change': 'Low Change (0-5%)',
            'moderate_change': 'Moderate Change (5-30%)',
            'high_change': 'High Change (≥30%)'
        }

        print("\n📊 Tile Categories:\n")

        for key, label in category_labels.items():
            refids = categories[key]
            count = len(refids)
            print(f"{label:30s}: {count:3d} tiles")

            if count > 0 and count <= 5:
                print(f"  REFIDs:")
                for refid in refids:
                    print(f"    - {refid}")
            elif count > 5:
                print(f"  First 3 REFIDs:")
                for refid in refids[:3]:
                    print(f"    - {refid}")
                print(f"    ... and {count - 3} more")
            print()

        # Save REFIDs to separate files
        reports_dir = Path(REPORTS_DIR)

        for key, refids in categories.items():
            output_file = reports_dir / f"refids_{key}.txt"
            with open(output_file, 'w') as f:
                for refid in refids:
                    f.write(f"{refid}\n")
            print(f"✓ Saved {len(refids)} REFIDs to: {output_file}")

        # Modeling recommendations
        print("\n" + "=" * 80)
        print("\n💡 Modeling Recommendations:\n")

        total_tiles = sum(len(refids) for refids in categories.values())
        zero_pct = (len(categories['zero_change']) / total_tiles) * 100 if total_tiles > 0 else 0
        low_pct = (len(categories['low_change']) / total_tiles) * 100 if total_tiles > 0 else 0

        if zero_pct > 10:
            print(f"⚠️  {zero_pct:.0f}% of tiles have ZERO change.")
            print(f"   Consider excluding these from training to focus on change detection.")
        elif zero_pct > 0:
            print(f"✓ Only {zero_pct:.0f}% of tiles have zero change - good for training.")

        if low_pct > 30:
            print(f"⚠️  {low_pct:.0f}% of tiles have LOW change (<5%).")
            print(f"   Use weighted loss functions (Focal Loss) to handle class imbalance.")
        elif low_pct > 0:
            print(f"✓ {low_pct:.0f}% of tiles have low change - manageable with weighted loss.")

        if len(categories['moderate_change']) > 0 and len(categories['high_change']) > 0:
            print(f"✓ Good distribution of moderate ({len(categories['moderate_change'])}) " +
                  f"and high change ({len(categories['high_change'])}) tiles.")
            print(f"  Stratified sampling recommended for train/val/test split.")

        # Calculate overall imbalance
        df_valid = df[df['change_ratio'].notna()]
        if len(df_valid) > 0:
            avg_change = df_valid['change_ratio'].mean()
            print(f"\n📈 Overall average change: {avg_change:.2f}%")

            if avg_change < 10:
                print(f"   Dataset is HIGHLY IMBALANCED (avg change < 10%)")
                print(f"   Recommendations:")
                print(f"     - Use Focal Loss or Dice Loss")
                print(f"     - Oversample change patches during training")
                print(f"     - Evaluate with F1-score, IoU (not accuracy)")
            elif avg_change < 30:
                print(f"   Dataset is MODERATELY IMBALANCED (10% ≤ avg change < 30%)")
                print(f"   Recommendations:")
                print(f"     - Use weighted Binary Cross-Entropy")
                print(f"     - Consider Focal Loss for better performance")
                print(f"     - Evaluate with F1, IoU, precision-recall")
            else:
                print(f"   Dataset is RELATIVELY BALANCED (avg change ≥ 30%)")
                print(f"   Standard loss functions should work well")

        print("\n" + "=" * 80)
        print("\n✅ Edge case identification complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
