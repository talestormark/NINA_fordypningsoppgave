"""
Script to extract and validate REFID identifiers from all data sources
"""

import sys
import re
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR


def extract_refid(filename):
    """
    Extract REFID from filename using regex pattern

    Filenames appear to be in format: a0-XXXXX_YY-ZZZZZ_<type>.tif
    We'll extract the ID portion before the final underscore and file type

    Args:
        filename: Name of the file

    Returns:
        REFID string or None if not found
    """
    # First try the expected REFID_XXX pattern
    pattern_refid = r'(REFID_\d+)'
    match = re.search(pattern_refid, filename)
    if match:
        return match.group(1)

    # If not found, extract the coordinate-based ID pattern
    # Pattern: extract everything before the underscore followed by known suffixes
    # e.g., "a0-07602270798631_51-64536656448906_mask.tif" -> "a0-07602270798631_51-64536656448906"
    # e.g., "a0-07602270798631_51-64536656448906_RGBY_Mosaic.tif" -> "a0-07602270798631_51-64536656448906"
    pattern_coords = r'^(.+?)_(?:RGBNIRRSWIRQ_Mosaic|RGBQ_Mosaic|RGBY_Mosaic|VEY_Mosaic|mask)\.tif$'
    match = re.search(pattern_coords, filename)
    if match:
        return match.group(1)

    return None


def get_refids_by_folder(data_root):
    """
    Extract unique REFIDs from each folder (excluding PlanetScope)

    Args:
        data_root: Path to root data directory

    Returns:
        dict: {folder_name: set_of_refids}
    """
    data_root = Path(data_root)
    refid_dict = {}

    print("\nExtracting REFIDs from each data source...")
    print("=" * 70)

    for key, folder_name in FOLDERS.items():
        # Skip PlanetScope as it has 1,962 images (not all match the 55 REFIDs)
        if key == 'planetscope':
            print(f"⊘  {folder_name:20s} - SKIPPED (contains all 1,962 images)")
            continue

        folder_path = data_root / folder_name

        if not folder_path.exists():
            print(f"⚠️  {folder_name:20s} - FOLDER NOT FOUND")
            refid_dict[folder_name] = set()
            continue

        # Extract REFIDs from all .tif files
        refids = set()
        for tif_file in folder_path.glob("*.tif"):
            refid = extract_refid(tif_file.name)
            if refid:
                refids.add(refid)

        refid_dict[folder_name] = refids
        print(f"✓  {folder_name:20s} - {len(refids):3d} unique REFIDs")

    print("=" * 70)

    return refid_dict


def find_common_refids(refid_dict):
    """
    Find intersection of REFIDs across all folders

    Args:
        refid_dict: Dictionary of {folder_name: set_of_refids}

    Returns:
        Sorted list of common REFIDs
    """
    if not refid_dict:
        return []

    # Start with first set
    common_refids = None
    for refids in refid_dict.values():
        if common_refids is None:
            common_refids = refids.copy()
        else:
            common_refids = common_refids.intersection(refids)

    return sorted(list(common_refids))


def save_refid_list(refids, output_path):
    """Save REFID list to text file (one per line)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for refid in refids:
            f.write(f"{refid}\n")

    print(f"\n✓ REFID list saved to: {output_path}")


def save_refid_presence_csv(refid_dict, common_refids, output_path):
    """Create CSV showing which REFIDs are present in which folders"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique REFIDs across all folders
    all_refids = set()
    for refids in refid_dict.values():
        all_refids.update(refids)

    # Create DataFrame
    data = []
    for refid in sorted(all_refids):
        row = {'refid': refid}
        for folder_name, refids in refid_dict.items():
            # Create column name from folder name
            col_name = f"in_{folder_name.lower().replace('_', '')}"
            row[col_name] = refid in refids
        data.append(row)

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ REFID presence CSV saved to: {output_path}")

    return df


if __name__ == "__main__":
    try:
        # Extract REFIDs from each folder
        refid_dict = get_refids_by_folder(DATA_ROOT)

        # Find common REFIDs
        common_refids = find_common_refids(refid_dict)

        # Display results
        print(f"\n📊 REFID Analysis:")
        print(f"   Common REFIDs across all sources: {len(common_refids)}")

        if common_refids:
            print(f"\n   First 5 REFIDs:")
            for refid in common_refids[:5]:
                print(f"      - {refid}")

        # Validation check
        if len(common_refids) != 55:
            print(f"\n⚠️  WARNING: Expected 55 common REFIDs, but found {len(common_refids)}")
        else:
            print(f"\n✅ Validation passed: Found exactly 55 common REFIDs")

        # Save results
        reports_dir = Path(REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save REFID list
        refid_list_file = reports_dir / "refid_list.txt"
        save_refid_list(common_refids, refid_list_file)

        # Save presence CSV
        presence_csv_file = reports_dir / "refid_presence.csv"
        df = save_refid_presence_csv(refid_dict, common_refids, presence_csv_file)

        # Show presence summary
        print(f"\n📋 REFID Presence Summary:")
        for folder_name, refids in refid_dict.items():
            col_name = f"in_{folder_name.lower().replace('_', '')}"
            if col_name in df.columns:
                count = df[col_name].sum()
                print(f"   {folder_name:20s}: {count:3d} REFIDs")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
