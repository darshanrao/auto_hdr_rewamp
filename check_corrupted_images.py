#!/usr/bin/env python3
"""Check automatic-lens-correction dataset for corrupted image files."""

import cv2
from pathlib import Path
from tqdm import tqdm
import sys

# Default paths - check both possible locations
DATA_ROOTS = [
    Path("/root/projects/automatic-lens-correction/lens-correction-train-cleaned"),
    Path("/dev/shm/automatic-lens-correction/lens-correction-train-cleaned"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_corrupted(path: Path) -> bool:
    """Try to load image; return True if corrupted."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            return True  # cv2 failed to decode
        # Quick sanity check: image has valid shape
        if img.size == 0:
            return True
        return False
    except Exception:
        return True


def main():
    data_root = None
    for root in DATA_ROOTS:
        if root.exists():
            data_root = root
            break

    if data_root is None:
        print("Error: Dataset not found. Checked:")
        for r in DATA_ROOTS:
            print(f"  - {r}")
        sys.exit(1)

    print(f"Scanning: {data_root}\n")

    files = sorted(p for p in data_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    corrupted = []
    for path in tqdm(files, desc="Checking images"):
        if is_corrupted(path):
            corrupted.append(path.name)

    # Report
    print(f"\nTotal images checked: {len(files)}")
    print(f"Corrupted: {len(corrupted)}")

    if corrupted:
        print("\nCorrupted files:")
        for name in corrupted[:50]:  # Show first 50
            print(f"  {name}")
        if len(corrupted) > 50:
            print(f"  ... and {len(corrupted) - 50} more")
    else:
        print("\nAll images OK.")

    return 0 if not corrupted else 1


if __name__ == "__main__":
    sys.exit(main())
