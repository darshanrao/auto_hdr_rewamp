#!/usr/bin/env python3
"""
Run inference with guided filter for edge alignment.

Same as inference_submit.py but applies guided filter after warping:
uses the original (distorted) image as guide to align edges in the
corrected output. Requires opencv-contrib-python:

    pip install opencv-contrib-python
"""
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_geoproj import GeoProjFlowNet, apply_flow

# Hardcoded paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _SCRIPT_DIR / "checkpoints" / "best_model.pth"
_TEST_DIR = Path("/root/projects/automatic-lens-correction/test-originals")
_OUTPUT_DIR = _SCRIPT_DIR / "submission_images_guided"
_ZIP_PATH = _SCRIPT_DIR / "submission_guided.zip"

IMG_SIZE = 256  # Must match training resolution

# Guided filter params: radius=8, eps=0.01 = balanced start
# Try radius=4, eps=0.001 for aggressive | radius=16, eps=0.05 for gentle
GUIDED_RADIUS = 8
GUIDED_EPS = 0.01


def tta_horizontal_flip(model, img_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Test-time augmentation: average flow from original + horizontally flipped.
    Exploits radial symmetry of barrel distortion for more accurate edge alignment.

    img_t: tensor (1, 3, H, W) normalized to [-1, 1]
    """
    model.eval()
    with torch.no_grad():
        flow_orig = model(img_t)

        img_flip = torch.flip(img_t, dims=[3])
        flow_flip = model(img_flip)

        flow_flip = torch.flip(flow_flip, dims=[3])
        flow_flip[:, 0, :, :] *= -1

        flow_avg = (flow_orig + flow_flip) / 2.0
    return flow_avg


def _has_ximgproc() -> bool:
    """Check if opencv-contrib ximgproc is available."""
    return hasattr(cv2, "ximgproc")


def guided_edge_align(
    corrected: np.ndarray,
    original: np.ndarray,
    radius: int = 8,
    eps: float = 0.01,
) -> np.ndarray:
    """
    Use original image to align edges in corrected output.

    corrected: warped output (H, W, 3) uint8 BGR
    original:  input distorted image (H, W, 3) uint8 BGR, same size
    radius:    filter radius; larger=more smoothing, smaller=more local
    eps:       regularization; smaller=sharper edges, larger=more smoothing
    """
    orig_f = original.astype(np.float32) / 255.0
    corr_f = corrected.astype(np.float32) / 255.0

    gf = cv2.ximgproc.createGuidedFilter(orig_f, radius, eps)
    result = gf.filter(corr_f)
    return (result * 255).clip(0, 255).astype(np.uint8)


def upsample_flow(flow_low: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Upsample flow from (IMG_SIZE x IMG_SIZE) to (target_h, target_w), scaling displacement magnitude."""
    flow_up = F.interpolate(
        flow_low, size=(target_h, target_w), mode="bilinear", align_corners=True
    )
    scale_x = target_w / float(IMG_SIZE)
    scale_y = target_h / float(IMG_SIZE)
    flow_up[:, 0] *= scale_x
    flow_up[:, 1] *= scale_y
    return flow_up


def run_inference(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    radius: int = GUIDED_RADIUS,
    eps: float = GUIDED_EPS,
    use_tta: bool = True,
    use_guided: bool = True,
):
    use_guided_filter = use_guided and _has_ximgproc()
    if use_guided and not _has_ximgproc():
        print("Warning: cv2.ximgproc not found, skipping guided filter. pip install opencv-contrib-python")

    model = GeoProjFlowNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval().to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])

    for p in tqdm(paths, desc="Inference"):
        img = cv2.imread(str(p))
        if img is None:
            continue

        image_id = p.stem
        h, w = img.shape[:2]
        is_portrait = h > w

        if is_portrait:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

        # Model runs at IMG_SIZE x IMG_SIZE
        img_in = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_in_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_in_rgb).float().permute(2, 0, 1) / 127.5 - 1.0
        t = t.unsqueeze(0).to(device)

        if use_tta:
            flow_low = tta_horizontal_flip(model, t, device)
        else:
            with torch.no_grad():
                flow_low = model(t)

        # Upsample flow and apply to original
        flow_full = upsample_flow(flow_low, h, w)
        img_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_full).float().permute(2, 0, 1) / 127.5 - 1.0
        img_t = img_t.unsqueeze(0).to(device)
        corrected_t = apply_flow(img_t, flow_full)

        # Denormalize to 0-255 BGR
        corrected = (
            (corrected_t[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5
        ).clip(0, 255).astype("uint8")
        corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

        # Guided filter: use original as guide to align edges in corrected (optional)
        if use_guided_filter:
            corrected = guided_edge_align(
                corrected=corrected,
                original=img,
                radius=radius,
                eps=eps,
            )

        if is_portrait:
            corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)

        out_name = f"{image_id}.jpg"
        cv2.imwrite(str(output_dir / out_name), corrected)

    return len(paths)


def create_submission_zip(output_dir: Path, zip_path: Path):
    """Zip corrected images as {image_id}.jpg (flat, no subdirs)."""
    jpg_files = list(output_dir.glob("*.jpg"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in tqdm(jpg_files, desc="Zipping"):
            zf.write(f, f.name)
    print(f"Created {zip_path} ({len(jpg_files)} images)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=GUIDED_RADIUS, help="Guided filter radius")
    parser.add_argument("--eps", type=float, default=GUIDED_EPS, help="Guided filter eps")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA horizontal flip")
    parser.add_argument("--no-guided", action="store_true", help="Disable guided filter")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")
    if not _TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {_TEST_DIR}")

    n = run_inference(
        _MODEL_PATH, _TEST_DIR, _OUTPUT_DIR, device,
        radius=args.radius, eps=args.eps,
        use_tta=not args.no_tta,
        use_guided=not args.no_guided,
    )
    print(f"Corrected {n} images -> {_OUTPUT_DIR}")
    print(f"TTA: {not args.no_tta} | Guided filter: {not args.no_guided} (r={args.radius}, eps={args.eps})")

    create_submission_zip(_OUTPUT_DIR, _ZIP_PATH)
    print(f"\nSubmission zip: {_ZIP_PATH}")
    print("Ready to upload.")


if __name__ == "__main__":
    main()
