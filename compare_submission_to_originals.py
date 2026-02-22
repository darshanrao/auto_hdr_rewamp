#!/usr/bin/env python3
"""
Compare submission (corrected) images vs original test images.
Reports pixel difference, SSIM, edge similarity to diagnose correction strength.
If training went well: corrected ≠ original (visible change from distortion fix).
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

TEST_DIR = Path("/root/projects/automatic-lens-correction/test-originals")
SUBMISSION_DIR = Path("/root/projects/auto_hdr_rewamp/submission_images")


def get_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sx**2 + sy**2)


def main():
    test_files = sorted(TEST_DIR.glob("*.jpg"))
    results = []

    for tp in tqdm(test_files, desc="Comparing"):
        sp = SUBMISSION_DIR / tp.name
        if not sp.exists():
            results.append({"name": tp.name, "status": "MISSING"})
            continue

        orig = cv2.imread(str(tp))
        sub = cv2.imread(str(sp))
        if orig is None or sub is None:
            results.append({"name": tp.name, "status": "READ_FAIL"})
            continue

        if orig.shape != sub.shape:
            results.append({"name": tp.name, "status": "SHAPE_MISMATCH"})
            continue

        # Mean absolute pixel difference (0-255 scale)
        diff = cv2.absdiff(orig, sub)
        mean_diff = np.mean(diff)
        mean_diff_per_channel = np.mean(diff, axis=(0, 1))

        # Edge map difference (how much edges moved)
        edge_orig = get_edges(orig)
        edge_sub = get_edges(sub)
        edge_diff = np.mean(np.abs(edge_orig - edge_sub))

        # Center vs corner (barrel: corners change more)
        h, w = orig.shape[:2]
        cy, cx = h // 2, w // 2
        margin = min(h, w) // 8
        center_diff = np.mean(diff[cy - margin : cy + margin, cx - margin : cx + margin])
        corner_regions = [
            diff[:margin, :margin],
            diff[:margin, -margin:],
            diff[-margin:, :margin],
            diff[-margin:, -margin:],
        ]
        corner_diff = np.mean([np.mean(r) for r in corner_regions])

        results.append({
            "name": tp.name,
            "status": "ok",
            "mean_diff": mean_diff,
            "center_diff": center_diff,
            "corner_diff": corner_diff,
            "edge_diff": edge_diff,
            "barrel_ratio": corner_diff / (center_diff + 1e-6),
        })

    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        print("No valid pairs found.")
        return

    mean_diffs = [r["mean_diff"] for r in ok]
    center_diffs = [r["center_diff"] for r in ok]
    corner_diffs = [r["corner_diff"] for r in ok]
    barrel_ratios = [r["barrel_ratio"] for r in ok]

    print("\n" + "=" * 60)
    print("SUBMISSION vs ORIGINAL TEST IMAGES")
    print("=" * 60)
    print(f"Pairs compared: {len(ok)} / {len(test_files)}")
    print()
    print("Mean pixel difference (0-255 scale):")
    print(f"  Mean:   {np.mean(mean_diffs):.2f}")
    print(f"  Median: {np.median(mean_diffs):.2f}")
    print(f"  Min:    {np.min(mean_diffs):.2f}")
    print(f"  Max:    {np.max(mean_diffs):.2f}")
    print()
    print("Center vs corner (barrel pattern: corners >> center):")
    print(f"  Center mean diff: {np.mean(center_diffs):.2f}")
    print(f"  Corner mean diff: {np.mean(corner_diffs):.2f}")
    print(f"  Barrel ratio (corner/center): {np.mean(barrel_ratios):.2f}")
    print()
    print("Interpretation:")
    print("  - Training data overall_diff: mild 0.5-2, normal 2-20, heavy >20")
    print("  - If mean_diff very low (<2): model may be under-correcting")
    print("  - If mean_diff in 2-15 range: visible correction, similar to train")
    print("  - Barrel ratio > 1.2: correction follows expected radial pattern")


if __name__ == "__main__":
    main()
