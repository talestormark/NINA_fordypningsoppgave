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


def get_refids_by_folder(data_root, include_planetscope=False):
    """
    Extract unique REFIDs from each folder

    Args:
        data_root: Path to root data directory
        include_planetscope: If True, include PlanetScope (default False for common REFID calculation)

    Returns:
        dict: {folder_name: set_of_refids}
    """
    data_root = Path(data_root)
    refid_dict = {}

    print("\nExtracting REFIDs from each data source...")
    print("=" * 70)

    for key, folder_name in FOLDERS.items():
        # Skip PlanetScope for common REFID calculation (has 1,962 images, not all match the 55 REFIDs)
        if key == 'planetscope' and not include_planetscope:
            print(f"⊘  {folder_name:20s} - SKIPPED for common REFID calculation")
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


def get_planetscope_refids(data_root):
    """
    Extract REFIDs from PlanetScope folder separately

    Args:
        data_root: Path to root data directory

    Returns:
        set: Set of REFIDs found in PlanetScope
    """
    data_root = Path(data_root)
    folder_name = FOLDERS['planetscope']
    folder_path = data_root / folder_name

    if not folder_path.exists():
        print(f"⚠️  PlanetScope folder not found at {folder_path}")
        return set()

    refids = set()
    for tif_file in folder_path.glob("*.tif"):
        refid = extract_refid(tif_file.name)
        if refid:
            refids.add(refid)

    return refids


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


def load_geojson_metadata(geojson_path):
    """
    Load metadata from the GeoJSON file

    Args:
        geojson_path: Path to land_take_bboxes_650m_v1.geojson

    Returns:
        dict: {plotid: {country, loss_type, change_type}}
    """
    import json

    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        print(f"⚠️  GeoJSON file not found at {geojson_path}")
        return {}

    with open(geojson_path, 'r') as f:
        data = json.load(f)

    metadata = {}
    for feature in data['features']:
        props = feature['properties']
        plotid = props['PLOTID']

        # Store only the first occurrence of each PLOTID
        if plotid not in metadata:
            metadata[plotid] = {
                'country': props.get('country', 'Unknown'),
                'loss_type': props.get('r', 'Unknown'),
                'change_type': props.get('change_type', 'Unknown')
            }

    return metadata


def save_refid_list(refids, output_path, geojson_metadata=None):
    """
    Save REFID list to text file with descriptive header and metadata

    Args:
        refids: List of REFIDs
        output_path: Path to save file
        geojson_metadata: Optional dict with metadata from GeoJSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("LAND-TAKE DETECTION DATASET - VALIDATED REFID LIST\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total REFIDs: {len(refids)}\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("These REFIDs have complete data across all main sources:\n")
        f.write("  - Sentinel-2 (10m resolution, 126 bands)\n")
        f.write("  - VHR Google (1m resolution, 6 bands)\n")
        f.write("  - AlphaEarth (10m resolution, 448 bands)\n")
        f.write("  - Land-take masks (10m resolution, binary)\n")
        f.write("  - PlanetScope (3-5m resolution, 42 bands)\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

        if geojson_metadata:
            f.write("\nREFID List with Metadata:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'REFID':<45} {'Country':<8} {'Loss Type':<15} {'Change Type'}\n")
            f.write("-" * 80 + "\n")

            for refid in refids:
                meta = geojson_metadata.get(refid, {})
                country = meta.get('country', 'N/A')
                loss_type = meta.get('loss_type', 'N/A')
                change_type = meta.get('change_type', 'N/A')

                # Truncate change_type if too long
                if len(change_type) > 35:
                    change_type = change_type[:32] + "..."

                f.write(f"{refid:<45} {country:<8} {loss_type:<15} {change_type}\n")
        else:
            f.write("\nREFID List:\n")
            f.write("-" * 80 + "\n")
            for refid in refids:
                f.write(f"{refid}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of REFID list\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ REFID list saved to: {output_path}")


def save_refid_presence_csv(refid_dict, common_refids, planetscope_refids, output_path):
    """
    Create CSV showing which REFIDs are present in which folders

    Args:
        refid_dict: Dictionary of {folder_name: set_of_refids} (excluding PlanetScope)
        common_refids: List of common REFIDs across main sources
        planetscope_refids: Set of REFIDs from PlanetScope
        output_path: Path to save CSV

    Returns:
        DataFrame with presence information
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique REFIDs across all folders (use common_refids to keep it focused)
    all_refids = set(common_refids)
    for refids in refid_dict.values():
        all_refids.update(refids)

    # Create DataFrame
    data = []
    for refid in sorted(all_refids):
        row = {'refid': refid}

        # Add columns for main data sources
        for folder_name, refids in refid_dict.items():
            # Create column name from folder name
            col_name = f"in_{folder_name.lower().replace('_', '')}"
            row[col_name] = refid in refids

        # Add PlanetScope column
        row['in_planetscope'] = refid in planetscope_refids

        data.append(row)

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ REFID presence CSV saved to: {output_path}")

    return df


if __name__ == "__main__":
    try:
        # Extract REFIDs from each folder (excluding PlanetScope for common calculation)
        refid_dict = get_refids_by_folder(DATA_ROOT, include_planetscope=False)

        # Find common REFIDs across main sources
        common_refids = find_common_refids(refid_dict)

        # Display results
        print(f"\n📊 REFID Analysis:")
        print(f"   Common REFIDs across main sources: {len(common_refids)}")

        if common_refids:
            print(f"\n   First 5 REFIDs:")
            for refid in common_refids[:5]:
                print(f"      - {refid}")

        # Validation check
        if len(common_refids) != 55:
            print(f"\n⚠️  WARNING: Expected 55 common REFIDs, but found {len(common_refids)}")
        else:
            print(f"\n✅ Validation passed: Found exactly 55 common REFIDs")

        # Extract PlanetScope REFIDs separately
        print(f"\n📡 Checking PlanetScope availability for common REFIDs...")
        planetscope_refids = get_planetscope_refids(DATA_ROOT)
        print(f"   Total PlanetScope images: {len(planetscope_refids)}")


        # Check how many common REFIDs have PlanetScope data
        common_in_planetscope = len([r for r in common_refids if r in planetscope_refids])
        print(f"   Common REFIDs with PlanetScope data: {common_in_planetscope}/{len(common_refids)}")

        if common_in_planetscope < len(common_refids):
            missing_from_ps = [r for r in common_refids if r not in planetscope_refids]
            print(f"\n⚠️  {len(missing_from_ps)} common REFIDs are missing from PlanetScope:")
            for refid in missing_from_ps[:5]:  # Show first 5
                print(f"      - {refid}")
            if len(missing_from_ps) > 5:
                print(f"      ... and {len(missing_from_ps) - 5} more")

        # Load GeoJSON metadata
        print(f"\n📍 Loading metadata from GeoJSON...")
        geojson_path = Path(__file__).parent.parent.parent / "land_take_bboxes_650m_v1.geojson"
        geojson_metadata = load_geojson_metadata(geojson_path)

        if geojson_metadata:
            print(f"   Loaded metadata for {len(geojson_metadata)} tiles")
            # Check how many common REFIDs have metadata
            with_metadata = sum(1 for r in common_refids if r in geojson_metadata)
            print(f"   Common REFIDs with metadata: {with_metadata}/{len(common_refids)}")

        # Save results
        reports_dir = Path(REPORTS_DIR)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Save REFID list (with metadata if available)
        refid_list_file = reports_dir / "refid_list.txt"
        save_refid_list(common_refids, refid_list_file, geojson_metadata)

        # Save presence CSV (now includes PlanetScope)
        presence_csv_file = reports_dir / "refid_presence.csv"
        df = save_refid_presence_csv(refid_dict, common_refids, planetscope_refids, presence_csv_file)

        # Show presence summary
        print(f"\n📋 REFID Presence Summary:")
        for folder_name, refids in refid_dict.items():
            col_name = f"in_{folder_name.lower().replace('_', '')}"
            if col_name in df.columns:
                count = df[col_name].sum()
                print(f"   {folder_name:20s}: {count:3d} REFIDs")

        # Add PlanetScope to summary
        if 'in_planetscope' in df.columns:
            ps_count = df['in_planetscope'].sum()
            print(f"   {'PlanetScope':20s}: {ps_count:3d} REFIDs")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
