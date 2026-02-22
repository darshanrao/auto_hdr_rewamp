#!/usr/bin/env python3
"""
Run inference on test images and create submission zip.
Output: corrected images named {image_id}.jpg, zipped as submission.zip

Flow is predicted at 512x512 (matches training), then upscaled and applied to
FULL-RESOLUTION image to preserve sharpness (no blur from upscaling a low-res result).
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
_MODEL_PATH = _SCRIPT_DIR / "checkpoints" / "best_model_36.06.pth"
_TEST_DIR = Path("/root/projects/automatic-lens-correction/test-originals")
_OUTPUT_DIR = _SCRIPT_DIR / "submission_images"
_ZIP_PATH = _SCRIPT_DIR / "submission.zip"


IMG_SIZE = 256  # Must match training resolution


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


def run_inference(model_path: Path, input_dir: Path, output_dir: Path, device: torch.device):
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

        image_id = p.stem  # e.g. 008195cc-eadd-42bb-99b9-108deb738154_g0
        h, w = img.shape[:2]
        is_portrait = h > w

        if is_portrait:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

        # Model runs at IMG_SIZE x IMG_SIZE (matches training)
        img_in = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_in_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_in_rgb).float().permute(2, 0, 1) / 127.5 - 1.0
        t = t.unsqueeze(0).to(device)

        with torch.no_grad():
            flow_low = model(t)

        # Upsample flow to full res and apply to ORIGINAL image (preserves sharpness)
        flow_full = upsample_flow(flow_low, h, w)
        img_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_full).float().permute(2, 0, 1) / 127.5 - 1.0
        img_t = img_t.unsqueeze(0).to(device)
        corrected_t = apply_flow(img_t, flow_full)
        corrected = (
            (corrected_t[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5
        ).clip(0, 255).astype("uint8")
        corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")
    if not _TEST_DIR.exists():
        raise FileNotFoundError(f"Test dir not found: {_TEST_DIR}")

    n = run_inference(_MODEL_PATH, _TEST_DIR, _OUTPUT_DIR, device)
    print(f"Corrected {n} images -> {_OUTPUT_DIR}")

    create_submission_zip(_OUTPUT_DIR, _ZIP_PATH)
    print(f"\nSubmission zip: {_ZIP_PATH}")
    print("Ready to upload.")


if __name__ == "__main__":
    main()
