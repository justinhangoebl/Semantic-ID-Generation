"""
Remaps item_id column in a semids CSV from original item IDs to sequential row-based IDs (1-indexed).

Usage:
    python remap_itemids.py <input_file> [output_file]

If output_file is omitted, writes to <input_stem>_remapped.csv in the same directory.
"""

import sys
import csv
from pathlib import Path


def remap_itemids(input_path: Path, output_path: Path) -> None:
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "item_id" not in fieldnames:
        raise ValueError(f"Expected 'item_id' column, got: {fieldnames}")

    for row_idx, row in enumerate(rows, start=1):
        row["item_id"] = row_idx

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Remapped {len(rows)} rows -> {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_stem(input_path.stem + "_remapped")

    remap_itemids(input_path, output_path)


if __name__ == "__main__":
    main()
