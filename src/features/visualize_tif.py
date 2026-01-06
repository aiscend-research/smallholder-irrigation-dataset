#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Show NDVI and RGB grids for BEFORE (*_image.tif) and AFTER (*_label.tif).

- NDVI is read directly from the NDVI band (index from JSON "bands").
- RGB is composed from B4/B3/B2, scaled 0..1 with per-step percentile stretch.
- Pixels with NO_DATA are shown as pure white in RGB (so AFTER masking is obvious).
"""

import os, glob, json
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Add uid as command line arg
import sys

FEATURES_DIR = "data/features"
SAVE_DIR = os.path.join(FEATURES_DIR, "visualization")
os.makedirs(SAVE_DIR, exist_ok=True)

NO_DATA = -9999
# Percentile stretch for RGB (per step, per channel)
RGB_P_LO, RGB_P_HI = 2, 98


def find_pair_files(unique_id: int):
    """
    Return tuple:
      (after_tif, after_json, before_tif, before_json)

    New naming (preferred):
      - AFTER  (masked stack):     *_image.tif / *_image.json
      - BEFORE (unmasked stack):   *_unmasked.tif / *_unmasked.json

    """
    # Find AFTER / masked (*_image.tif)
    after_tif = None
    for p in glob.glob(os.path.join(FEATURES_DIR, f"{unique_id}_*_image.tif")):
        after_tif = p
        break
    if not after_tif:
        return None, None, None, None
    after_json = after_tif.replace("_image.tif", "_image.json")

    # Preferred BEFORE / unmasked
    before_tif = after_tif.replace("_image.tif", "_unmasked.tif")
    before_json = after_tif.replace("_image.tif", "_unmasked.json")

    if not os.path.exists(before_tif) or not os.path.exists(before_json):
        # Legacy fallback to *_label.*
        legacy_tif = after_tif.replace("_image.tif", "_label.tif")
        legacy_json = after_tif.replace("_image.tif", "_label.json")
        if os.path.exists(legacy_tif) and os.path.exists(legacy_json):
            before_tif, before_json = legacy_tif, legacy_json
        else:
            before_tif = None
            before_json = None

    return after_tif, after_json, before_tif, before_json


def read_stack(tif_path, json_path):
    """Read GeoTIFF and reshape to (T,B,H,W) using the JSON's shape."""
    with rasterio.open(tif_path) as src:
        raw = src.read()  # (B*T, H, W)
    with open(json_path, "r") as f:
        meta = json.load(f)
    T, B, H, W = meta["shape"]
    bands = meta["bands"]
    stack = raw.reshape(B, T, H, W).transpose(1, 0, 2, 3)  # -> (T,B,H,W)
    return stack, bands, meta


# NDVI plotting
def plot_ndvi_grid_from_band(stack, bands, title, save_path, nodata=NO_DATA):
    """Plot NDVI directly from stack[:, NDVI, ...]."""
    try:
        ndvi_idx = bands.index("NDVI")
    except ValueError:
        raise RuntimeError(f"NDVI not found in bands: {bands}")

    T = stack.shape[0]
    cols = 6
    rows = int(np.ceil(T / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.ravel()

    for t in range(T):
        ndvi_raw = stack[t, ndvi_idx]
        ndvi = ndvi_raw.astype(np.float32) / 10000.0
        ndvi[ndvi_raw == nodata] = np.nan
        axes[t].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
        axes[t].set_title(f"Step {t+1}", fontsize=8)
        axes[t].axis("off")

    for t in range(T, len(axes)):
        axes[t].axis("off")

    fig.suptitle(title, fontsize=20, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# RGB helpers & plotting
def _stretch_01(channel_int, p_lo=2, p_hi=98):
    """
    Percentile stretch of a single channel (int16 reflectance 0..10000).
    Returns float32 0..1; NO_DATA left as NaN for now.
    """
    ch = channel_int.astype(np.float32) / 10000.0
    mask = (channel_int == NO_DATA)
    vals = ch[~mask]
    if vals.size == 0:
        out = np.zeros_like(ch, dtype=np.float32)
        out[mask] = np.nan
        return out

    lo, hi = np.percentile(vals, (p_lo, p_hi))
    if hi <= lo + 1e-6:
        # Avoid divide-by-zero
        out = np.clip(ch, 0.0, 1.0)
    else:
        out = (ch - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0)
    out[mask] = np.nan
    return out


def make_rgb_image(step_cube, bands):
    """
    Compose an RGB image (H,W,3) from a single timestep cube (B,H,W) using B4/B3/B2.
    - Applies percentile stretch per channel.
    - Converts NO_DATA to white.
    """
    try:
        r_idx = bands.index("B4")
        g_idx = bands.index("B3")
        b_idx = bands.index("B2")
    except ValueError:
        missing = [b for b in ("B4", "B3", "B2") if b not in bands]
        raise RuntimeError(f"Missing RGB bands in stack: {missing}")

    r = _stretch_01(step_cube[r_idx], RGB_P_LO, RGB_P_HI)
    g = _stretch_01(step_cube[g_idx], RGB_P_LO, RGB_P_HI)
    b = _stretch_01(step_cube[b_idx], RGB_P_LO, RGB_P_HI)

    rgb = np.stack([r, g, b], axis=-1)  # (H,W,3)
    # Any NaN (masked / NO_DATA) -> white
    nan_mask = np.isnan(rgb).any(axis=-1)
    if nan_mask.any():
        rgb[nan_mask] = 1.0
    return rgb.astype(np.float32)


def plot_rgb_grid(stack, bands, title, save_path):
    """Plot RGB grid using B4/B3/B2; masked pixels appear white."""
    T = stack.shape[0]
    cols = 6
    rows = int(np.ceil(T / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.ravel()

    for t in range(T):
        rgb = make_rgb_image(stack[t], bands)
        axes[t].imshow(rgb)
        axes[t].set_title(f"Step {t+1}", fontsize=8)
        axes[t].axis("off")

    for t in range(T, len(axes)):
        axes[t].axis("off")

    fig.suptitle(title, fontsize=20, y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# Main
if __name__ == "__main__":
    uid = None

    if len(sys.argv) > 1:
        try:
            uid = int(sys.argv[1])
        except ValueError:
            raise SystemExit("Usage: python3 visualize_tif.py [uid]")
    else:
        uid = None

    after_tif, after_json, before_tif, before_json = find_pair_files(uid)
    if not after_tif:
        raise SystemExit(f"No *_image.tif found for uid={uid} in {FEATURES_DIR}")

    # BEFORE (unmasked) — from *_unmasked.*
    if not before_tif or not before_json:
        print(f"[Warn] No BEFORE stack found for uid={uid} (looked for *_unmasked.tif/json")
    else:
        stack_bef, bands_bef, _ = read_stack(before_tif, before_json)
        plot_ndvi_grid_from_band(
            stack_bef, bands_bef,
            f"NDVI Before Masking (UID {uid})",
            os.path.join(SAVE_DIR, f"uid{uid}_ndvi_before.png"),
        )
        plot_rgb_grid(
            stack_bef, bands_bef,
            f"RGB Before Masking (UID {uid})",
            os.path.join(SAVE_DIR, f"uid{uid}_rgb_before.png"),
        )

    # AFTER (masked) — from *_image.*
    stack_aft, bands_aft, _ = read_stack(after_tif, after_json)
    plot_ndvi_grid_from_band(
        stack_aft, bands_aft,
        f"NDVI After Masking (UID {uid})",
        os.path.join(SAVE_DIR, f"uid{uid}_ndvi_after.png"),
    )
    plot_rgb_grid(
        stack_aft, bands_aft,
        f"RGB After Masking (UID {uid})",
        os.path.join(SAVE_DIR, f"uid{uid}_rgb_after.png"),
    )