#!/usr/bin/env python3
"""
GeoProj Lens Distortion Correction — Full Training Script
Single run on GPU. Uses modelNetM (EncoderNet + DecoderNet) to predict flow field.

Paths (hardcoded):
  CSV:       final_clean_dataset.csv (next to this script)
  Images:    /root/projects/automatic-lens-correction

Usage:
  python train_geoproj.py
  python train_geoproj.py --batch_size 64 --workers 16   # RTX 5090: larger batch, max workers

Defaults: batch_size=32, workers=16, lr=1e-4

  # Inference
  python train_geoproj.py --mode inference --input_dir test_images --output_dir corrected
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional: skimage for SSIM (channel_axis vs multichannel)
try:
    from skimage.metrics import structural_similarity
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False


# =============================================================================
# GeoProj Model (modelNetM)
# =============================================================================

class PlainEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual)


class PlainDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = (
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            if stride == 1
            else nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = (
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            if stride == 1
            else nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, output_padding=1),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual)


def _make_encoder_layer(block, in_ch, out_ch, n_blocks, stride):
    layers = [block(in_ch, out_ch, stride=stride)]
    for _ in range(1, n_blocks):
        layers.append(block(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


def _make_decoder_layer(block, in_ch, out_ch, n_blocks, stride):
    layers = [block(in_ch, in_ch, stride=1) for _ in range(n_blocks - 1)]
    layers.append(block(in_ch, out_ch, stride=stride))
    return nn.Sequential(*layers)


class EncoderNet(nn.Module):
    def __init__(self, layers=(1, 1, 1, 1, 2)):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.en_layer1 = _make_encoder_layer(PlainEncoderBlock, 64, 64, layers[0], stride=1)
        self.en_layer2 = _make_encoder_layer(ResEncoderBlock, 64, 128, layers[1], stride=2)
        self.en_layer3 = _make_encoder_layer(ResEncoderBlock, 128, 256, layers[2], stride=2)
        self.en_layer4 = _make_encoder_layer(ResEncoderBlock, 256, 512, layers[3], stride=2)
        self.en_layer5 = _make_encoder_layer(ResEncoderBlock, 512, 512, layers[4], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.en_layer1(x)
        x = self.en_layer2(x)
        x = self.en_layer3(x)
        x = self.en_layer4(x)
        x = self.en_layer5(x)
        return x


class DecoderNet(nn.Module):
    def __init__(self, layers=(1, 1, 1, 1, 2)):
        super().__init__()
        self.de_layer5 = _make_decoder_layer(ResDecoderBlock, 512, 512, layers[4], stride=2)
        self.de_layer4 = _make_decoder_layer(ResDecoderBlock, 512, 256, layers[3], stride=2)
        self.de_layer3 = _make_decoder_layer(ResDecoderBlock, 256, 128, layers[2], stride=2)
        self.de_layer2 = _make_decoder_layer(ResDecoderBlock, 128, 64, layers[1], stride=2)
        self.de_layer1 = _make_decoder_layer(PlainDecoderBlock, 64, 64, layers[0], stride=1)
        self.conv_end = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.de_layer5(x)
        x = self.de_layer4(x)
        x = self.de_layer3(x)
        x = self.de_layer2(x)
        x = self.de_layer1(x)
        return self.conv_end(x)


class GeoProjFlowNet(nn.Module):
    """Encoder + Decoder predicting flow field (2×H×W)."""

    def __init__(self):
        super().__init__()
        self.encoder = EncoderNet([1, 1, 1, 1, 2])
        self.decoder = DecoderNet([1, 1, 1, 1, 2])

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =============================================================================
# Flow Warping
# =============================================================================

def apply_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp img by flow. Flow (B,2,H,W): dx, dy.
    For each output pixel (y,x), sample img at (y+dy, x+dx).
    img, flow: range [-1, 1] for img; flow in pixels.
    """
    B, C, H, W = img.shape
    device = img.device

    yy = torch.arange(H, dtype=torch.float32, device=device)
    xx = torch.arange(W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    # Sampling coordinates: (x+dx, y+dy) for each pixel
    grid_x_new = (grid_x + flow[:, 0]).clamp(0, W - 1.001)
    grid_y_new = (grid_y + flow[:, 1]).clamp(0, H - 1.001)

    # grid_sample expects normalized coords in [-1, 1]
    grid_norm = torch.stack(
        [
            2.0 * grid_x_new / (W - 1) - 1.0,
            2.0 * grid_y_new / (H - 1) - 1.0,
        ],
        dim=-1,
    )
    return F.grid_sample(img, grid_norm, mode="bilinear", padding_mode="border", align_corners=True)


# =============================================================================
# Loss Functions
# =============================================================================

def gradient_loss(pred_corrected: torch.Tensor, target_corrected: torch.Tensor) -> torch.Tensor:
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(pred_corrected.device)
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(pred_corrected.device)
    )

    def to_gray(img):
        return img[:, 0:1] * 0.299 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.114

    pg = to_gray(pred_corrected)
    tg = to_gray(target_corrected)

    pgx = F.conv2d(pg, sobel_x, padding=1)
    pgy = F.conv2d(pg, sobel_y, padding=1)
    tgx = F.conv2d(tg, sobel_x, padding=1)
    tgy = F.conv2d(tg, sobel_y, padding=1)

    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def flow_endpoint_loss(pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
    diff = pred_flow - gt_flow
    epe = torch.sqrt((diff**2).sum(dim=1) + 1e-6)
    return epe.mean()


def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx, dy = flow[:, 0], flow[:, 1]
    return (
        torch.abs(dx[:, :, 1:] - dx[:, :, :-1]).mean()
        + torch.abs(dx[:, 1:, :] - dx[:, :-1, :]).mean()
        + torch.abs(dy[:, :, 1:] - dy[:, :, :-1]).mean()
        + torch.abs(dy[:, 1:, :] - dy[:, :-1, :]).mean()
    )


def photometric_loss(pred_corrected: torch.Tensor, target_corrected: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_corrected, target_corrected)


def total_loss(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    dist_img: torch.Tensor,
    corr_img: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
):
    pred_corrected = apply_flow(dist_img, pred_flow)

    l_edge = gradient_loss(pred_corrected, corr_img)
    l_flow = flow_endpoint_loss(pred_flow, gt_flow)
    l_photo = photometric_loss(pred_corrected, corr_img)
    l_smooth = flow_smoothness_loss(pred_flow)

    loss = 1.00 * l_edge + 0.70 * l_flow + 0.30 * l_photo + 0.05 * l_smooth

    if sample_weights is not None:
        loss = loss * sample_weights.mean()

    return loss, {
        "edge": l_edge.item(),
        "flow": l_flow.item(),
        "photo": l_photo.item(),
        "smooth": l_smooth.item(),
    }


# =============================================================================
# Validation Metrics
# =============================================================================

def compute_epe(pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> float:
    diff = pred_flow - gt_flow
    epe = torch.sqrt((diff**2).sum(dim=1) + 1e-6)
    return epe.mean().item()


def compute_edge_similarity(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    def get_edges(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sx**2 + sy**2)
        return edge / (edge.max() + 1e-6)

    pred_edges = get_edges(pred_np)
    true_edges = get_edges(gt_np)
    return float(1.0 - np.abs(pred_edges - true_edges).mean())


def compute_ssim(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    if not _SSIM_AVAILABLE:
        return 0.0
    try:
        return float(structural_similarity(pred_np, gt_np, data_range=255, channel_axis=2))
    except TypeError:
        return float(structural_similarity(pred_np, gt_np, data_range=255, multichannel=True))


def compute_line_straightness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.5

    deviations = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        dev = min(angle, abs(90 - angle), abs(180 - angle))
        deviations.append(dev)

    mean_dev = np.mean(deviations)
    return float(max(0.0, 1.0 - (mean_dev / 45.0)))


# =============================================================================
# Dataset
# =============================================================================

class LensCorrectionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 256,
        augment: bool = False,
        data_root: str | Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.data_root = Path(data_root) if data_root else None

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if self.data_root is not None:
            path = self.data_root / path.name
        return path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dist_path = self._resolve_path(row["dist_path"])
        corr_path = self._resolve_path(row["corr_path"])
        is_portrait = bool(row["is_portrait"])
        weight = float(row["weight"])

        dist = cv2.imread(str(dist_path))
        corr = cv2.imread(str(corr_path))

        if dist is None or corr is None:
            raise FileNotFoundError(f"Missing: {dist_path} or {corr_path}")

        if is_portrait:
            dist = cv2.rotate(dist, cv2.ROTATE_90_CLOCKWISE)
            corr = cv2.rotate(corr, cv2.ROTATE_90_CLOCKWISE)

        dist = cv2.resize(dist, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        corr = cv2.resize(corr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Augmentation: same transform for both
        do_flip = False
        if self.augment:
            do_flip = random.random() < 0.5
            if do_flip:
                dist = cv2.flip(dist, 1)
                corr = cv2.flip(corr, 1)
            alpha = random.uniform(0.9, 1.1)
            dist = np.clip(dist.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            corr = np.clip(corr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        # GT flow: backward flow (corr -> dist). Sample dist at (x+dx, y+dy) to get corr(x,y)
        gray_dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_corr = cv2.cvtColor(corr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            gray_corr,
            gray_dist,
            None,
            pyr_scale=0.5,
            levels=5,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow = flow.transpose(2, 0, 1).astype(np.float32)

        if do_flip:
            flow = np.flip(flow, axis=2).copy()
            flow[0] *= -1

        dist = cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)
        corr = cv2.cvtColor(corr, cv2.COLOR_BGR2RGB)

        dist_t = torch.from_numpy(dist).float().permute(2, 0, 1) / 127.5 - 1.0
        corr_t = torch.from_numpy(corr).float().permute(2, 0, 1) / 127.5 - 1.0
        flow_t = torch.from_numpy(flow)

        return dist_t, corr_t, flow_t, torch.tensor(weight, dtype=torch.float32)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001, save_path: str = "best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module, epoch: int):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f"  ✓ New best: {val_loss:.4f} → saved to {self.save_path}")
        else:
            self.counter += 1
            print(
                f"  No improvement for {self.counter}/{self.patience} epochs "
                f"(best={self.best_loss:.4f} at epoch {self.best_epoch + 1})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n  EARLY STOPPING at epoch {epoch + 1}")
                print(f"  Best epoch was: {self.best_epoch + 1}")
                print(f"  Best val loss:  {self.best_loss:.4f}")


# =============================================================================
# Training & Validation
# =============================================================================

def train_one_epoch(model, loader, optimizer, scaler, device, grad_clip: float = 1.0):
    model.train()
    running_loss = 0.0
    components = {"edge": 0.0, "flow": 0.0, "photo": 0.0, "smooth": 0.0}
    n = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for dist_t, corr_t, flow_t, weights in pbar:
        dist_t = dist_t.to(device)
        corr_t = corr_t.to(device)
        flow_t = flow_t.to(device)
        weights = weights.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred_flow = model(dist_t)
            loss, comp = total_loss(pred_flow, flow_t, dist_t, corr_t, weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = dist_t.shape[0]
        running_loss += loss.item() * bs
        for k in components:
            components[k] += comp[k] * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / n, {k: v / n for k, v in components.items()}


def validate(model, loader, device):
    model.eval()
    all_epe, all_edge, all_ssim, all_straight = [], [], [], []
    total_val_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for dist_t, corr_t, flow_t, weights in tqdm(loader, desc="Val", leave=False):
            dist_t = dist_t.to(device)
            corr_t = corr_t.to(device)
            flow_t = flow_t.to(device)
            weights = weights.to(device)

            pred_flow = model(dist_t)
            loss, _ = total_loss(pred_flow, flow_t, dist_t, corr_t, weights)
            total_val_loss += loss.item()
            n_batches += 1

            epe = compute_epe(pred_flow, flow_t)
            all_epe.append(epe)

            pred_corr = apply_flow(dist_t, pred_flow)

            for i in range(dist_t.shape[0]):
                pred_np = ((pred_corr[i].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                corr_np = ((corr_t[i].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                all_edge.append(compute_edge_similarity(pred_np, corr_np))
                all_ssim.append(compute_ssim(pred_np, corr_np))
                all_straight.append(compute_line_straightness(pred_np))

    return {
        "val_loss": total_val_loss / max(n_batches, 1),
        "epe": float(np.mean(all_epe)),
        "edge_sim": float(np.mean(all_edge)),
        "ssim": float(np.mean(all_ssim)),
        "straightness": float(np.mean(all_straight)),
    }


# =============================================================================
# Main
# =============================================================================

# Hardcoded paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_CSV_PATH = _SCRIPT_DIR / "final_clean_dataset.csv"
_DATA_ROOT = Path("/root/projects/automatic-lens-correction/lens-correction-train-cleaned")


def main(cmd_args=None):
    ap = argparse.ArgumentParser(description="GeoProj lens correction training")
    ap.add_argument("--batch_size", type=int, default=16)  # 16 for 512x512; 32 for 256x256
    ap.add_argument("--workers", type=int, default=16)     # match CPU cores for data loading
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)  # OLD: 3e-4 (was too aggressive)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--min_delta", type=float, default=0.001)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    ap.add_argument("--img_size", type=int, default=512, help="Training resolution (512 for better straightness)")
    args = ap.parse_args(cmd_args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"

    df = pd.read_csv(_CSV_PATH)
    df = df[df["use_in_train"] == True].copy()
    if len(df) == 0:
        raise SystemExit("No training samples (use_in_train=True) found in CSV")

    n_val = int(len(df) * args.val_ratio)
    n_train = len(df) - n_val
    indices = np.random.permutation(len(df))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Priority 2: heavy weight 5.0 -> 10.0; oversample heavy 5x
    if "category" in train_df.columns:
        heavy_mask = train_df["category"] == "heavy"
    else:
        heavy_mask = train_df["weight"] == 5.0
    train_df = train_df.copy()
    train_df.loc[heavy_mask, "weight"] = 10.0
    heavy_rows = train_df[heavy_mask]
    for _ in range(4):
        train_df = pd.concat([train_df, heavy_rows], ignore_index=True)

    train_ds = LensCorrectionDataset(
        train_df, img_size=args.img_size, augment=True, data_root=_DATA_ROOT
    )
    val_ds = LensCorrectionDataset(
        val_df, img_size=args.img_size, augment=False, data_root=_DATA_ROOT
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = GeoProjFlowNet()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    early_stop = EarlyStopping(patience=args.patience, min_delta=args.min_delta, save_path=str(best_path))

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(ckpt, strict=True)
        print(f"Resumed from {args.resume}")

    print(f"Train samples: {n_train}, Val samples: {n_val}")
    print(f"Device: {device}, GPUs: {torch.cuda.device_count()}")

    for epoch in range(args.epochs):
        train_loss, comp = train_one_epoch(model, train_dl, optimizer, scaler, device, args.grad_clip)
        metrics = validate(model, val_dl, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {metrics['val_loss']:.4f} | "
            f"EPE: {metrics['epe']:.3f}px | EdgeSim: {metrics['edge_sim']:.4f} | "
            f"SSIM: {metrics['ssim']:.4f} | Straight: {metrics['straightness']:.4f} | "
            f"LR: {lr:.2e} | "
            f"[E:{comp['edge']:.4f} F:{comp['flow']:.4f} P:{comp['photo']:.4f} S:{comp['smooth']:.4f}]"
        )

        early_stop.step(metrics["val_loss"], model.module if hasattr(model, "module") else model, epoch)
        if early_stop.should_stop:
            break

    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\nLoaded best model from epoch {early_stop.best_epoch + 1}, val_loss={early_stop.best_loss:.4f}")


def run_inference(
    model_path: str,
    input_dir: str,
    output_dir: str,
    device: torch.device | None = None,
    img_size: int = 512,
):
    """Run lens correction on images in input_dir, save to output_dir."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoProjFlowNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval().to(device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]

    for p in tqdm(paths, desc="Inference"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        is_portrait = h > w
        if is_portrait:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

        img_in = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_in).float().permute(2, 0, 1) / 127.5 - 1.0
        t = t.unsqueeze(0).to(device)

        with torch.no_grad():
            flow_low = model(t)

        # Upsample flow to full res and apply to original (preserves sharpness)
        flow_up = F.interpolate(
            flow_low, size=(h, w), mode="bilinear", align_corners=True
        )
        flow_up[:, 0] *= w / float(img_size)
        flow_up[:, 1] *= h / float(img_size)
        img_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_full).float().permute(2, 0, 1) / 127.5 - 1.0
        img_t = img_t.unsqueeze(0).to(device)
        corrected_t = apply_flow(img_t, flow_up)
        corrected = ((corrected_t[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

        if is_portrait:
            corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite(str(out_path / p.name), corrected)
    print(f"Saved {len(paths)} images to {output_dir}")


if __name__ == "__main__":
    top = argparse.ArgumentParser()
    top.add_argument("--mode", choices=["train", "inference"], default="train")
    top.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="For inference: path to best_model.pth")
    top.add_argument("--input_dir", type=str, default=None, help="For inference: folder of distorted images")
    top.add_argument("--output_dir", type=str, default="corrected_output", help="For inference: output folder")
    top_args, rest = top.parse_known_args()

    if top_args.mode == "inference":
        if not top_args.input_dir:
            raise SystemExit("--input_dir required for inference")
        run_inference(top_args.model, top_args.input_dir, top_args.output_dir)
    else:
        main(rest)
