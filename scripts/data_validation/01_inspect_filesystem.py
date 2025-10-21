"""
Script to inspect the filesystem and count files in each data folder
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_ROOT, FOLDERS, REPORTS_DIR


def list_data_folders(data_root):
    """
    List all subdirectories and count .tif files in each

    Args:
        data_root: Path to root data directory

    Returns:
        dict: {folder_name: file_count}
    """
    data_root = Path(data_root)
    results = {}

    print(f"\nInspecting data folder: {data_root}")
    print("=" * 70)

    if not data_root.exists():
        print(f"❌ ERROR: Data root does not exist: {data_root}")
        return results

    for key, folder_name in FOLDERS.items():
        folder_path = data_root / folder_name

        if not folder_path.exists():
            print(f"⚠️  {folder_name:20s} - FOLDER NOT FOUND")
            results[folder_name] = 0
            continue

        # Count .tif files
        tif_files = list(folder_path.glob("*.tif"))
        file_count = len(tif_files)
        results[folder_name] = file_count

        print(f"✓  {folder_name:20s} - {file_count:5d} .tif files")

    print("=" * 70)

    return results


def save_results(results, output_path):
    """Save folder structure results to text file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("Folder Structure Inspection Results\n")
        f.write("=" * 70 + "\n\n")

        for folder_name, count in results.items():
            f.write(f"{folder_name:20s}: {count:5d} files\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total folders: {len(results)}\n")
        f.write(f"Total files: {sum(results.values())}\n")

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    try:
        # Inspect data folders
        results = list_data_folders(DATA_ROOT)

        # Save results
        output_file = Path(REPORTS_DIR) / "folder_structure.txt"
        save_results(results, output_file)

        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total folders inspected: {len(results)}")
        print(f"   Total .tif files found: {sum(results.values())}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
