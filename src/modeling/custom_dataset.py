#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom dataset and data utilities for Sentinel-2 multi-temporal crop classification.
Includes:
- Torch Dataset for multi-temporal Sentinel-2 cubes
- Helper functions for loading and flattening raster data from manifest
- Visualization utilities for predictions and masks
"""

import os
import json
import re
import glob
import torch
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------------------------------------------------------
# Sentinel-2 Band Info
# ----------------------------------------------------------------------
S2_BAND_NAMES = [
    "B2 (Blue)",
    "B3 (Green)",
    "B4 (Red)",
    "B5 (Vegetation Red Edge 1)",
    "B6 (Vegetation Red Edge 2)",
    "B7 (Vegetation Red Edge 3)",
    "B8 (Near Infrared, NIR)",
    "B8A (Narrow NIR)",
    "B11 (Short Wave Infrared 1, SWIR 1)",
    "B12 (Short Wave Infrared 2, SWIR 2)",
    "NDVI (Normalized Difference Vegetation Index)",
    "EVI (Enhanced Vegetation Index)",
    "NDWI (Normalized Difference Water Index)",
    "SCL (Scene Classification Layer)",
]
SHORT_BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8",
    "B8A", "B11", "B12", "NDVI", "EVI", "NDWI", "SCL"
]

S2_BAND_NAME_TO_INDEX = {}
for i, name in enumerate(S2_BAND_NAMES):
    short = name.split(" ")[0].split("(")[0].strip()
    S2_BAND_NAME_TO_INDEX[short] = i
    S2_BAND_NAME_TO_INDEX[name] = i


def get_band_indices(band_names):
    """Convert a list of band names or indices to a list of integer indices."""
    indices = []
    for b in band_names:
        if isinstance(b, int):
            indices.append(b)
        elif isinstance(b, str):
            idx = S2_BAND_NAME_TO_INDEX.get(b)
            if idx is None:
                raise ValueError(f"Unknown band name: {b}")
            indices.append(idx)
        else:
            raise ValueError(f"Band identifier must be str or int, got {type(b)}")
    return indices


# ----------------------------------------------------------------------
# Dataset + Helper Functions
# ----------------------------------------------------------------------
def load_dataset_from_manifest(stems: list[str], manifest_df: pd.DataFrame, label_bands) -> list:
    """
    Load image/label data directly from absolute paths in CV manifest.
    Handles single or multiple label bands automatically.
    Returns a list of (image_array, label_array, stem) tuples.
    """
    manifest_index = manifest_df.set_index("stem")
    dataset = []
    logger.info(f"[load] Loading {len(stems)} samples directly from manifest paths...")

    # --- Handle label_bands argument ---
    if isinstance(label_bands, int):
        label_bands = list(range(1, label_bands + 1))
    elif isinstance(label_bands, list) and len(label_bands) == 1 and label_bands[0] == 1:
        logger.warning("[load] Detected single label band [1]; expanding to 8 default label bands.")
        label_bands = list(range(1, 9))

    missing_in_manifest = 0
    missing_files = 0
    failed_reads = 0

    for i, stem in enumerate(stems):
        if stem not in manifest_index.index:
            missing_in_manifest += 1
            logger.warning(f"[load] Stem '{stem}' not found in manifest, skipping")
            continue

        row = manifest_index.loc[stem]
        img_path = Path(str(row["image_path"]))
        lab_path = Path(str(row["label_path"]))

        if not img_path.exists():
            missing_files += 1
            logger.warning(f"[load] Image missing: {img_path}")
            continue
        if not lab_path.exists():
            missing_files += 1
            logger.warning(f"[load] Label missing: {lab_path}")
            continue

        try:
            with rasterio.open(img_path) as src:
                image = src.read()
            with rasterio.open(lab_path) as src:
                label = src.read(label_bands)
            dataset.append((image, label, stem))

            if (i + 1) % 50 == 0 or (i + 1) == len(stems):
                logger.info(f"[load] Loaded {len(dataset)}/{i + 1} samples")

        except Exception as e:
            failed_reads += 1
            logger.warning(f"[load] Failed to read {stem}: {e}")
            continue

    logger.info(f"[load] Successfully loaded {len(dataset)}/{len(stems)} samples")
    if missing_in_manifest > 0:
        logger.warning(f"[load] {missing_in_manifest} stems not found in manifest")
    if missing_files > 0:
        logger.warning(f"[load] {missing_files} files missing on disk")
    if failed_reads > 0:
        logger.warning(f"[load] {failed_reads} files failed to read")

    if len(dataset) == 0:
        raise RuntimeError("No valid samples were loaded from manifest.")

    # Final shape summary
    try:
        sample_image, sample_label, _ = dataset[0]
        logger.info(f"[load] Example image shape: {sample_image.shape}, label shape: {sample_label.shape}")
    except Exception:
        pass

    return dataset

def flatten_dataset_from_tuples(dataset: list, pixels_per_image: int = None) -> tuple:
    """
    Flatten dataset from list of (image, label, stem) tuples with optional pixel sampling.
    Logs dataset progress, sampling strategy, and final flattened shapes.
    """
    X_list, y_list, stems_list = [], [], []

    total_pixels_original = 0
    total_pixels_used = 0

    # Log start
    sampling_msg = (
        f"{pixels_per_image} pixels per image" if pixels_per_image else "using ALL pixels"
    )
    logger.info(f"[flatten] Flattening {len(dataset)} samples ({sampling_msg})...")

    # Iterate over dataset
    for idx, (image, label, stem) in enumerate(dataset):
        n_bands, height, width = image.shape
        total_pixels = height * width
        total_pixels_original += total_pixels

        X_full = image.reshape(n_bands, -1).T
        y_full = label.reshape(label.shape[0], -1).T

        # Optional sampling
        if pixels_per_image and total_pixels > pixels_per_image:
            sel = np.random.choice(total_pixels, pixels_per_image, replace=False)
            X_full = X_full[sel]
            y_full = y_full[sel]
            total_pixels_used += pixels_per_image
        else:
            total_pixels_used += total_pixels

        X_list.append(X_full)
        y_list.append(y_full)
        stems_list.append(stem)

        # Log progress every 100 images
        if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
            logger.info(
                f"[flatten] Processed {idx + 1}/{len(dataset)} samples "
                f"({total_pixels_used:,} / {total_pixels_original:,} pixels used)"
            )

    # Stack arrays
    X = np.vstack(X_list).astype(np.float32)
    y = np.vstack(y_list).astype(np.int8)

    # Final summary
    logger.info(f"[flatten] Final shapes: X={X.shape}, y={y.shape}")
    logger.info(f"[flatten] Total pixels used: {total_pixels_used:,} / {total_pixels_original:,}")

    return X, y, stems_list


def plot_predictions(dataset: list, model, num_samples: int = 2, save_path: str = None):
    """Visualize predictions vs ground truth for randomly chosen samples."""
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend to prevent Tkinter errors
    import matplotlib.pyplot as plt

    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(sample_indices):
        image, label, stem = dataset[sample_idx]
        n_bands, height, width = image.shape
        X_sample = image.reshape(n_bands, -1).T
        y_pred = model.predict(X_sample)
        y_pred_img = y_pred.reshape(height, width)

        # RGB visualization if 3+ bands
        if n_bands >= 3:
            rgb = np.stack([image[2], image[1], image[0]], axis=-1)
            rgb = np.clip(rgb / np.nanmax(rgb) * 255, 0, 255).astype(np.uint8)
            axes[idx, 0].imshow(rgb)
            axes[idx, 0].set_title(f"{stem}: RGB")
        else:
            axes[idx, 0].imshow(image[0], cmap="gray")

        axes[idx, 1].imshow(label[0], cmap="viridis")
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 2].imshow(y_pred_img, cmap="viridis")
        axes[idx, 2].set_title("Prediction")

        for ax in axes[idx]:
            ax.axis("off")

    plt.tight_layout()

    # Always save instead of showing
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"[viz] Saved visualization to {save_path}")
    else:
        # Default saving behavior if no path provided
        default_path = "default_visualization.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        logger.info(f"[viz] Saved visualization to {default_path}")

    plt.close(fig)


# ----------------------------------------------------------------------
# Torch Dataset Definition
# ----------------------------------------------------------------------
class MultiTemporalCropDataset(Dataset):
    """Dataset loader for multi-temporal Sentinel-2 crop data."""

    def __init__(
        self,
        image_dir=None,
        label_dir=None,
        data_dir=None,
        label_bands=None,
        image_band_names=None,
        time_step_selection=None,
        drop_cloud_images=True,
    ):
        if data_dir is not None:
            image_dir = label_dir = data_dir

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_band_names = image_band_names or S2_BAND_NAMES
        self.label_bands = label_bands or list(range(1, 9))
        self.num_bands = 14
        self.num_timesteps = 37
        self.drop_cloud_images = drop_cloud_images
        self.time_step_selection = time_step_selection

        # -----------------------------
        # Flexible matching logic
        # -----------------------------
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*_image.tif")))
        label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.tif")))

        def normalize_id(filename):
            stem = Path(filename).stem
            # remove _KL_label, _JL_label, etc.
            stem = re.sub(r'_[A-Za-z]+_label$', '', stem)
            stem = re.sub(r'_label$', '', stem)
            stem = re.sub(r'_image$', '', stem)
            return stem

        image_ids = {normalize_id(f) for f in image_files}
        label_ids = {normalize_id(f) for f in label_files}
        paired_ids = sorted(image_ids & label_ids)
        logger.info(f"[dataset] Matched {len(paired_ids)} paired samples")

        self.paired_image_files = [os.path.join(self.image_dir, f"{uid}_image.tif") for uid in paired_ids]
        # Find correct label variant (handles _KL_label, _JL_label)
        self.paired_mask_files = []
        for uid in paired_ids:
            pattern = os.path.join(self.label_dir, f"{uid}_*_label.tif")
            matches = glob.glob(pattern)
            if not matches:
                matches = glob.glob(os.path.join(self.label_dir, f"{uid}_label.tif"))
            if matches:
                self.paired_mask_files.append(matches[0])
            else:
                logger.warning(f"[dataset] No label found for {uid}")
                self.paired_mask_files.append(None)

        self.paired_unique_ids = [int(uid) if uid.isdigit() else i for i, uid in enumerate(paired_ids)]

    def __len__(self):
        return len(self.paired_image_files)

    def __getitem__(self, idx):
        image_path = self.paired_image_files[idx]
        mask_path = self.paired_mask_files[idx]
        unique_id = self.paired_unique_ids[idx]

        if mask_path is None:
            raise RuntimeError(f"No mask found for {image_path}")

        # Load image
        with rasterio.open(image_path) as src:
            arr = src.read()
            image_tensor = torch.from_numpy(arr).float()
            H, W = image_tensor.shape[1:]
            image_tensor = image_tensor.reshape(self.num_timesteps, self.num_bands, H, W).permute(1, 0, 2, 3)
            image_tensor = torch.where(image_tensor == -9999, torch.tensor(0.0), image_tensor)

        # Optional time averaging
        if self.time_step_selection is not None:
            selected = []
            for sel in self.time_step_selection:
                if isinstance(sel, int):
                    selected.append(image_tensor[:, sel:sel + 1, :, :])
                elif isinstance(sel, list):
                    selected.append(image_tensor[:, sel, :, :].mean(dim=1, keepdim=True))
            image_tensor = torch.cat(selected, dim=1)

        band_indices = get_band_indices(self.image_band_names)
        image_tensor = image_tensor[band_indices, ...]

        # Load label
        with rasterio.open(mask_path) as src:
            mask_array = src.read(self.label_bands)
            mask_tensor = torch.from_numpy(mask_array).float()
            mask_tensor = torch.where(mask_tensor == -9999, torch.tensor(0.0), mask_tensor)
            if mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor[0]

        return {"image": image_tensor, "mask": mask_tensor, "id": unique_id}
