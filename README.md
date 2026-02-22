# GeoProj: Flow-Based Lens Distortion Correction

End-to-end pipeline for correcting barrel distortion in images using a learned flow prediction network. This repository implements: data cleaning, training with GeoProjFlowNet, and inference. The approach is based on [*Blind Geometric Distortion Correction on Images Through Deep Learning*](https://arxiv.org/abs/1909.03459) (Li et al., 2019), which uses CNNs to predict a dense displacement field from distorted images for non-parametric correction.

## Overview

We predict a 2D dense flow field from a distorted image and apply it via bicubic warping to produce the corrected image. Following the GeoProj formulation, the flow represents where each pixel in the distorted image should move to obtain the corrected image. Training uses pairs of (distorted, corrected) images; ground-truth flow is computed with Farneback optical flow. The model is an encoder–decoder network (similar to GeoNetM in the paper) that outputs flow at low resolution; inference upsamples and applies it to full-resolution images.

**Key design choices:**
- **Flow prediction** (GeoProj-style): non-parametric displacement field handles barrel distortion and pipeline artifacts without fitting k1/k2.
- **Low-resolution flow** (256×256) for efficiency; bicubic warping at full resolution preserves sharpness.
- **Test-time augmentation** (horizontal flip averaging) exploits radial symmetry of barrel distortion.
- **Data cleaning** removes label noise (identical pairs, shift-contaminated images) to improve training.

---

## Repository Structure

```
auto_hdr_rewamp/
├── data_cleaning.ipynb         # Dataset cleaning pipeline
├── train_geoproj.py            # Training script (GeoProjFlowNet)
├── inference_submit_bicubic_tta.py   # Main inference (bicubic + TTA)
├── submit_and_score.py         # Upload zip, poll score
├── final_clean_dataset.csv     # Cleaned training metadata
├── full_dataset_clean.csv      # Intermediate cleaned data
├── shift_analysis.csv          # Shift detection results
├── check_corrupted_images.py   # Integrity checks
├── check_train_val_split.py    # Split validation
└── discarded_inference_pipelines/    # Alternative inference variants
```

---

## 1. Data Cleaning

The cleaning pipeline is documented in `data_cleaning.ipynb` and summarized in `Notes/Notes_data_cleaning:.ini`. Raw data: **23,118 image pairs** (distorted + corrected).

### Steps

| Step | What | Action |
|------|------|--------|
| 1 | Structure | Confirm dimensions, orientation; all pairs same size |
| 2 | Orientation | Keep portrait images; rotate before processing |
| 3 | Height groups | Main (h≈1366) and tall (h≈1534); both kept, same distortion physics |
| 4 | Center analysis | Center diff ≈ 1.0 → JPEG noise only; pure barrel distortion |
| 5 | Magnitude | Categorize by `overall_diff`: identical, mild, normal, heavy |
| 6 | Barrel ratio | `corner_diff/center_diff`; low ratio → suspicious (non-barrel) |
| 7 | Shift detection | Phase correlation; remove shifted pairs |

### Categories and Weights

- **Identical** (diff &lt; 0.5): 134 pairs → **REMOVED** (labels contradict other samples)
- **Mild** (0.5–2.0): 699 pairs → weight **0.3**
- **Normal** (2.0–20.0): 21,487 pairs → weight **1.0**
- **Heavy** (&gt; 20.0): 147 pairs → weight **5.0**, oversampled (critical for wide-angle)

### Shift Removal

Low barrel ratio indicates uniform change instead of radial pattern (typical of global translation). Phase correlation (`cv2.phaseCorrelate`) detects translation; **651 pairs** with shift &gt; 2 px are removed.

### Output

- **full_dataset_clean.csv** — after categorization and shift detection
- **final_clean_dataset.csv** — filtered with `use_in_train=True`, used for training

**Final training set:** 22,333 pairs.

---

## 2. Training

### Architecture (GeoProjFlowNet)

Our model follows the encoder–decoder design of [GeoProj](https://arxiv.org/abs/1909.03459): a network that maps from the image domain to the flow domain. As in the paper, the flow is a forward map (distorted → corrected); we use grid-sample-based resampling instead of the iterative search described in the paper.

- **Encoder:** 5 stages (64→128→256→512→512 channels), ResNet-style blocks
- **Decoder:** symmetric with transposed convolutions
- **Output:** 2-channel flow field (dx, dy) at input resolution

### Ground-Truth Flow

Computed with Farneback optical flow (`corr → dist`): for each pixel in the corrected image, flow gives the source location in the distorted image.

### Loss

```
total_loss = 1.00 × gradient_loss      (edge alignment)
            + 0.70 × flow_endpoint_loss (EPE)
            + 0.30 × photometric_loss
            + 0.05 × flow_smoothness
            + 0.10 × corner_regularization
```

**Corner regularization** penalizes large flow at corners to avoid overcorrection.

### Dataset

- **LensCorrectionDataset** loads `final_clean_dataset.csv`
- Images resized to 256×256 for training
- Portrait images rotated to landscape
- Augmentation: horizontal flip, brightness scaling (same on both images)
- Sample weights: mild 0.3, normal 1.0, heavy 10.0 (and oversampled)

### Usage

```bash
pip install -r requirements_train.txt
python train_geoproj.py --batch_size 96 --workers 16
```

Paths: `final_clean_dataset.csv`, images under `lens-correction-train-cleaned/` (set `_DATA_ROOT` in script).

---

## 3. Inference Pipeline

### Flow

1. Load image; if portrait, rotate 90° clockwise
2. Resize to 256×256, normalize to [-1, 1]
3. Run model (with TTA: average flow from original + flipped)
4. Upsample flow to full resolution (scale dx/dy by resolution ratio)
5. Warp full-res image with **bicubic** interpolation (`F.grid_sample`)
6. If portrait, rotate back

### Bicubic vs Bilinear

Bicubic warping reduces blurring at edges compared to bilinear.

### TTA (Test-Time Augmentation)

- Run model on image and on horizontally flipped image
- Average flows; flip averaged flow back and negate x-component
- Improves symmetry; ~2× inference cost

### Usage

```bash
python inference_submit_bicubic_tta.py
```

Outputs: `submission_images_bicubic_tta/`, `submission_bicubic_tta.zip`.

### Submission

```bash
python submit_and_score.py
```

Uploads zip to `bounty.autohdr.com`, polls for score.

---

## 4. File Conventions

| CSV Column | Meaning |
|------------|---------|
| `dist_path` | Path to distorted (or original capture) image |
| `corr_path` | Path to corrected (or generated) image |
| `overall_diff` | Mean pixel difference between pair |
| `barrel_ratio` | `corner_diff / center_diff` |
| `category` | `mild`, `normal`, `heavy` |
| `weight` | Sample weight for training |
| `use_in_train` | Whether to include in training |

---

## Dependencies

- Python 3.10+
- PyTorch ≥ 2.0
- OpenCV, NumPy, Pandas, tqdm, scikit-image

---

## Citation

If you use this code, please cite the GeoProj paper and this repository:

```bibtex
@article{li2019blind,
  title={Blind Geometric Distortion Correction on Images Through Deep Learning},
  author={Li, Xiaoyu and Zhang, Bo and Sander, Pedro V and Liao, Jing},
  journal={arXiv preprint arXiv:1909.03459},
  year={2019}
}
```
