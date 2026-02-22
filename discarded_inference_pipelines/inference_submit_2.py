#!/usr/bin/env python3
"""
Run inference on test images with spatial damping.
Same as inference_submit.py but applies spatial damping to reduce corner over-warping.

Flow is predicted at IMG_SIZE×IMG_SIZE, upsampled to full resolution, damped, then
applied to the ORIGINAL full-res image.
"""
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_geoproj import GeoProjFlowNet, apply_flow

# Hardcoded paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _SCRIPT_DIR / "checkpoints" / "best_model_36.06.pth"
_TEST_DIR = Path("/root/projects/automatic-lens-correction/test-originals")
_OUTPUT_DIR = _SCRIPT_DIR / "submission_images_2"
_ZIP_PATH = _SCRIPT_DIR / "submission_2.zip"

IMG_SIZE = 256  # Must match training resolution
DAMPING_STRENGTH = 0.85  # 0.90, 0.85, 0.80, 0.75 - lower = more damping at corners

# Per-worker state (used by multiprocessing)
_worker_model = None
_worker_device = None


def spatial_damping(flow: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Apply spatial damping to flow. Center keeps 100% of flow, corners keep strength%.
    Input: flow (B, 2, H, W), strength in [0, 1]
    """
    B, _, H, W = flow.shape
    device = flow.device

    # Normalized distance from center: 0 at center, 1 at corners
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    # Euclidean distance from center (normalized so corners ≈ 1)
    # Corner (1,1) has distance sqrt(2); normalize by sqrt(2) so max ≈ 1
    distance = torch.sqrt(grid_x ** 2 + grid_y ** 2) / (2 ** 0.5)
    distance = distance.clamp(0, 1)

    # Damping mask: center=1.0, corners=strength
    # mask = 1.0 - (1.0 - strength) * distance
    mask = 1.0 - (1.0 - strength) * distance
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return flow * mask


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


def _process_one_image(p: Path, output_dir: Path) -> int:
    """Process a single image. Uses global _worker_model and _worker_device (set by init)."""
    global _worker_model, _worker_device
    img = cv2.imread(str(p))
    if img is None:
        return 0

    image_id = p.stem
    h, w = img.shape[:2]
    is_portrait = h > w

    if is_portrait:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    img_in = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_in_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_in_rgb).float().permute(2, 0, 1) / 127.5 - 1.0
    t = t.unsqueeze(0).to(_worker_device)

    with torch.no_grad():
        flow_low = _worker_model(t)

    flow_full = upsample_flow(flow_low, h, w)
    flow_damped = spatial_damping(flow_full, DAMPING_STRENGTH)

    img_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_full).float().permute(2, 0, 1) / 127.5 - 1.0
    img_t = img_t.unsqueeze(0).to(_worker_device)
    corrected_t = apply_flow(img_t, flow_damped)

    corrected = (
        (corrected_t[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5
    ).clip(0, 255).astype("uint8")
    corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

    if is_portrait:
        corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)

    out_name = f"{image_id}.jpg"
    cv2.imwrite(str(output_dir / out_name), corrected)

    return 1


def _init_worker(model_path: Path, device: torch.device, num_threads: int = 1):
    """Load model once per worker process."""
    global _worker_model, _worker_device
    torch.set_num_threads(num_threads)
    _worker_device = device
    _worker_model = GeoProjFlowNet()
    _worker_model.load_state_dict(torch.load(model_path, map_location=device))
    _worker_model = _worker_model.eval().to(device)


def run_inference(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    verbose: bool = False,
    num_workers: int = 1,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])

    if num_workers <= 1:
        # Serial: load model once in main process
        global _worker_model, _worker_device
        _worker_device = device
        _worker_model = GeoProjFlowNet()
        _worker_model.load_state_dict(torch.load(model_path, map_location=device))
        _worker_model = _worker_model.eval().to(device)

        for i, p in enumerate(tqdm(paths, desc="Inference")):
            _process_one_image(p, output_dir)
            if verbose and i == 0:
                img = cv2.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"Original image: {w}×{h}")
    else:
        # Parallel: each worker loads model via initializer (1 thread each to avoid oversubscription)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=partial(_init_worker, model_path, device, 1),
        ) as ex:
            futures = [ex.submit(_process_one_image, p, output_dir) for p in paths]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
                f.result()

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
    parser.add_argument("--verbose", "-v", action="store_true", help="Print validation info for first image")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (when GPU busy with training)")
    parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel CPU workers (default 1, use 32 for multi-core)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")
    if not _TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {_TEST_DIR}")

    n = run_inference(_MODEL_PATH, _TEST_DIR, _OUTPUT_DIR, device, verbose=args.verbose, num_workers=args.workers)
    print(f"Corrected {n} images -> {_OUTPUT_DIR}")
    print(f"Damping strength: {DAMPING_STRENGTH}")

    create_submission_zip(_OUTPUT_DIR, _ZIP_PATH)
    print(f"\nSubmission zip: {_ZIP_PATH}")
    print("Ready to upload.")


if __name__ == "__main__":
    main()
