import os
# def test_flatten_dataset_with_dummy_data():
#     import torch
#
# NOTE: Setting per_band_time=True in flatten_dataset will flatten as one row per (band, time, pixel),
# filtering pixels only at that specific (band, time, pixel) location (not across all bands/times).

import numpy as np
from tqdm import tqdm
import torch

from sklearn.impute import SimpleImputer, KNNImputer
from typing import Optional, Tuple

# Time interpolation imputer
def _time_interp_row(row: np.ndarray, T: int, C: int, fill_constant: float = 0.0) -> np.ndarray:
    """
    Reshape a 1D feature vector (length = T*C) to (T, C), linearly interpolate along time
    for each band independently, and reshape back. NaNs are filled by:
      - all-NaN column -> fill_constant
      - single observed value -> broadcast across time
      - otherwise -> linear interpolation with end extrapolation
    """
    arr = row.reshape(T, C).astype(float)
    idx = np.arange(T)
    for c in range(C):
        col = arr[:, c]
        m = ~np.isnan(col)
        if not m.any():
            arr[:, c] = fill_constant
        elif m.sum() == 1:
            arr[:, c] = col[m][0]
        else:
            arr[:, c] = np.interp(idx, idx[m], col[m]) 
    return arr.reshape(T * C)


def time_interpolate_features(X: np.ndarray, T: int, C: int, fill_constant: float = 0.0) -> np.ndarray:
    """
    Apply temporal interpolation per row. X must have feature dimension T*C (bands stacked per timestep or vice versa,
    but consistent with how flatten_dataset outputs: features = T * C).
    """
    out = np.empty_like(X, dtype=float)
    for i in tqdm(range(X.shape[0]), desc="Imputing (time interp)", unit="row"):
        out[i] = _time_interp_row(X[i], T, C, fill_constant=fill_constant)
    return out

def flatten_dataset(dataset, ignore_value_in_image=None, debug=True, per_band_time=False):
    """
    Flattens a multi-temporal crop dataset for ML.
    If per_band_time=True, returns one row per (band, time, pixel).
    If False, returns one row per (x, y) location, using the old logic.
    Args:
        dataset: PyTorch Dataset where each sample is a dict:
            'image': Tensor (C, T, H, W)
            'mask' : Tensor (H, W) or (B, H, W)
        ignore_value_in_image: Optional value in image pixels to ignore (e.g., -9999 for clouds)
        debug: If True, print per-sample debug info.
        per_band_time: If True, use new flattening logic (one row per (band, time, pixel))
    Returns:
        X: np.ndarray
        y: np.ndarray
    Notes:
        - Only filters out pixels for which the value at (band, time, x, y) is NaN/ignore_value_in_image.
    """
    import torch

    if not per_band_time:
        # Original logic (row per x,y)
        X_list = []
        y_list = []
        n_samples = 0
        n_skipped = 0

        for idx, sample in enumerate(tqdm(dataset, desc="Flattening dataset")):
            image = sample['image']  # (C, T, H, W)
            mask = sample['mask']    # (H, W) or (B, H, W)

            if image.ndim != 4:
                raise ValueError(f"Expected image shape (C, T, H, W), got {image.shape}")
            if mask.ndim not in (2, 3):
                raise ValueError(f"Expected mask shape (H, W) or (B, H, W), got {mask.shape}")

            C, T, H, W = image.shape
            image = image.permute(2, 3, 1, 0)  # (H, W, T, C)
            image_flat = image.reshape(H * W, T * C)

            # Replace -9999 with nan by default
            image_flat = image_flat.clone()
            image_flat[image_flat == -9999] = float('nan')

            if ignore_value_in_image is not None:
                image_flat[image_flat == ignore_value_in_image] = float('nan')

            if mask.ndim == 2:
                mask_flat = mask.reshape(H * W, 1)  # (N, 1) for consistency
            else:
                B = mask.shape[0]
                mask_flat = mask.permute(1, 2, 0).reshape(H * W, B)  # (N, B)

            mask_np = mask_flat.numpy()
            image_np = image_flat.numpy()
            valid = np.ones(mask_np.shape[0], dtype=bool)

            X_valid = image_np[valid]
            y_valid = mask_np[valid]

            if X_valid.shape[0] == 0:
                n_skipped += 1
            else:
                X_list.append(X_valid)
                y_list.append(y_valid)
                n_samples += X_valid.shape[0]

        if not X_list:
            X = np.empty((0, T * C))
            y = np.empty((0, mask_flat.shape[1]))
        else:
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)

        # If only one mask band, squeeze to (N,)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        return X, y

    # per (band, time, pixel)
    X_list = []
    y_list = []
    
    total_pixels = 0
    valid_pixels = 0

    for i, sample in enumerate(tqdm(dataset, desc="Flattening dataset")):
        image = sample['image']  # (C, T, H, W)
        mask = sample['mask']    # (H, W) or (B, H, W)

        C, T, H, W = image.shape
        image_np = image.numpy()
        if mask.ndim == 2:
            mask_np = mask.numpy().reshape(1, H, W)  # (1, H, W)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        image_np = image_flat.numpy()
        mask_np = mask_flat.numpy()
        
        # Check for extreme values that might indicate invalid data
        if ignore_value_in_image is not None:
            image_invalid = (image_np == ignore_value_in_image)
            if np.any(image_invalid):
                print(f"Sample {i}: Found {np.sum(image_invalid)} pixels with ignore value {ignore_value_in_image}")
        
        # Check for NaN values
        mask_nan = np.isnan(mask_np)
        image_nan = np.isnan(image_np)
        
        # Count total and invalid pixels
        sample_total = H * W
        sample_invalid = np.sum(np.any(mask_nan, axis=1) | np.any(image_nan, axis=1))
        sample_valid = sample_total - sample_invalid
        
        total_pixels += sample_total
        valid_pixels += sample_valid
        
        print(f"Sample {i}: {sample_valid}/{sample_total} valid pixels ({sample_valid/sample_total*100:.1f}%)")
        
        # If all pixels are invalid, skip this sample
        if sample_valid == 0:
            print(f"Warning: Sample {i} has no valid pixels, skipping")
            continue

        # Filter out invalid pixels
        valid = ~(np.any(mask_nan, axis=1) | np.any(image_nan, axis=1))
        
        X_valid = image_np[valid]
        y_valid = mask_np[valid]

        X_list.append(X_valid)
        y_list.append(y_valid)

    if not X_list:
        raise ValueError("No valid pixels found in any sample! Check your data for NaN or invalid values.")
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"Total: {valid_pixels}/{total_pixels} valid pixels ({valid_pixels/total_pixels*100:.1f}%)")
    print(f"Final shapes: X={X.shape}, y={y.shape}")

    # If only one mask band, squeeze to (N,)
    if y.shape[1] == 1:
        y = y.squeeze(1)
    return X, y


# def test_flatten_dataset_with_dummy_data():
#     import torch

#     # Test single-band masks (all samples have mask shape (H, W))
#     sample1 = {
#         'image': torch.randn(3, 4, 5, 5),
#         'mask': torch.randint(0, 2, (5, 5)).float()
#     }
#     dataset1 = [sample1, sample1]
#     X1, y1 = flatten_dataset(dataset1, ignore_value_in_image=None, debug=False)
#     print("\nTest: single-band mask")
#     print(f"X shape: {X1.shape}, y shape: {y1.shape}")
#     print(f"First 5 rows of y1:\n{y1[:5]}")

#     # Test multi-band masks (all samples have mask shape (B, H, W))
#     sample2 = {
#         'image': torch.randn(3, 4, 5, 5),
#         'mask': torch.randint(0, 2, (2, 5, 5)).float()
#     }
#     dataset2 = [sample2, sample2]
#     X2, y2 = flatten_dataset(dataset2, ignore_value_in_image=None, debug=False)
#     print("\nTest: multi-band mask")
#     print(f"X shape: {X2.shape}, y shape: {y2.shape}")
#     print(f"First 5 rows of y2:\n{y2[:5]}")


# def test_flatten_dataset_per_band_time_edge_cases():
#     print("\n==== Testing flatten_dataset (per_band_time=True) ====")
#     H, W = 4, 5  # Small for quick tests

#     # (1) Basic case: No -9999, 1 band, 2 time, 1 mask band
#     sample = {
#         'image': torch.ones(1, 2, H, W),
#         'mask': torch.zeros(H, W)
#     }
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=True)
#     assert X.shape[0] == 1 * 2 * H * W, f"Expected {1*2*H*W}, got {X.shape[0]}"
#     assert y.shape[0] == X.shape[0]
#     print("[PASSED] Simple (no -9999, single band/time)")

#     # (2) -9999 in one time/band, only that one pixel should be filtered
#     image = torch.ones(1, 2, H, W)
#     image[0, 1, 1, 2] = -9999
#     sample = {'image': image, 'mask': torch.zeros(H, W)}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=True)
#     assert X.shape[0] == 1 * 2 * H * W - 1, "[FAILED] -9999 pixel not filtered right"
#     print("[PASSED] -9999 pixel masked correctly for per_band_time")

#     # (3) Multiple bands, time, mask bands
#     image = torch.ones(3, 3, H, W)
#     mask = torch.randint(0, 2, (2, H, W)).float()
#     image[2, 2, 0, 0] = -9999  # One missing pixel
#     sample = {'image': image, 'mask': mask}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=True)
#     total_pixels = 3 * 3 * H * W * 2  # bands * times * h * w * mask bands
#     # There are two mask bands. For each, one pixel is filtered, so subtract 2.
#     expected = total_pixels - 2
#     print(f"[DEBUG] (Multiple bands/time/mask bands)")
#     print(f"  Image shape: {image.shape}, Mask shape: {mask.shape}")
#     print(f"  Expected rows: {expected}")
#     print(f"  Actual rows: {X.shape[0]}")
#     print(f"  -9999 pixels (should be 2): {np.sum(image.numpy() == -9999)}")
#     assert X.shape[0] == expected, "[FAILED] Multiple bands/time/mask bands"
#     print("[PASSED] Multiple bands/time/mask bands")

#     # (4) All -9999 in one band/time: that slice gone, others remain
#     image = torch.ones(1, 2, H, W)
#     image[0, 0, :, :] = -9999  # All pixels in first time, gone
#     sample = {'image': image, 'mask': torch.zeros(H, W)}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=True)
#     assert X.shape[0] == 1 * 1 * H * W, "[FAILED] All pixels in one timepoint gone"
#     print("[PASSED] All -9999 in one timepoint: filtered right")

#     # (5) All -9999: X and y are empty
#     image = torch.full((1, 2, H, W), -9999.)
#     sample = {'image': image, 'mask': torch.zeros(H, W)}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=True)
#     assert X.size == 0 and y.size == 0, "[FAILED] All -9999 should result in empty arrays"
#     print("[PASSED] All -9999 returns empty arrays")

# def test_flatten_dataset_original_edge_cases():
#     print("\n==== Testing flatten_dataset (original per-pixel logic) ====")
#     H, W = 3, 3

#     # (1) All ones, no -9999
#     image = torch.ones(2, 2, H, W)
#     mask = torch.zeros(H, W)
#     sample = {'image': image, 'mask': mask}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=False)
#     assert X.shape[0] == H * W, "[FAILED] All pixels should remain"
#     print("[PASSED] Simple: no filtering")

#     # (2) -9999 in just one band/t, whole pixel filtered for all bands/times
#     image = torch.ones(2, 2, H, W)
#     image[1, 0, 1, 1] = -9999
#     sample = {'image': image, 'mask': torch.zeros(H, W)}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=False)
#     assert (X.shape[0] == H * W - 1), "[FAILED] Entire pixel should be filtered"
#     print("[PASSED] -9999 in any band/time => pixel filtered")

#     # (3) Mask is NaN for some pixel, that pixel filtered
#     mask = torch.zeros(H, W)
#     mask[2, 1] = float('nan')
#     image = torch.ones(2, 2, H, W)
#     sample = {'image': image, 'mask': mask}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=False)
#     assert (X.shape[0] == H * W - 1), "[FAILED] NaN in mask should be filtered"
#     print("[PASSED] NaN in mask => pixel filtered")

#     # (4) All -9999: everything filtered, output empty
#     image = torch.full((2, 2, H, W), -9999.)
#     sample = {'image': image, 'mask': torch.zeros(H, W)}
#     dataset = [sample]
#     X, y = flatten_dataset(dataset, ignore_value_in_image=-9999, per_band_time=False)
#     assert X.size == 0 and y.size == 0, "[FAILED] All -9999 should result in empty arrays"
#     print("[PASSED] All -9999: empty output")

# if __name__ == "__main__":
#     test_flatten_dataset_per_band_time_edge_cases()
#     test_flatten_dataset_original_edge_cases()

def _write_nan_table_txt(path_txt: str, counts_ct: np.ndarray):
    """
    Save a pretty text table: one row per band, 37 integers per row (t0..t36).
    """
    C, T = counts_ct.shape
    with open(path_txt, "w") as f:
        f.write("Number of NaN pixels per band/timepoint:\n")
        for b in range(C):
            row = "  ".join(f"{int(c):5d}" for c in counts_ct[b])
            f.write(f"Band {b+1:2d}: {row}\n")

def _save_nan_counts(out_dir: str, basename: str, counts_ct: np.ndarray):
    """
    Save counts_ct (C,T) to TXT and CSV with the same base name.
    """
    os.makedirs(out_dir, exist_ok=True)
    # TXT
    _write_nan_table_txt(os.path.join(out_dir, f"{basename}.txt"), counts_ct)
    # CSV
    np.savetxt(os.path.join(out_dir, f"{basename}.csv"), counts_ct, delimiter=",", fmt="%d")

def compute_nan_stats_for_dataset(dataset, out_dir: str, split_name: str = "train", save_per_sample: bool = False):
    """
    Compute NaN counts per (band, time) for each sample in a dataset and an aggregate over the split.

    Assumes each dataset item is a dict with 'image' tensor of shape (C, T, H, W),
    where cloud/missing pixels have already been set to NaN (as in MultiTemporalCropDataset).
    Saves an aggregate TXT/CSV into out_dir. If save_per_sample is True, per-sample TXT/CSV files are also saved.
    The aggregate file is always saved.
    """
    if len(dataset) == 0:
        os.makedirs(out_dir, exist_ok=True)
        return

    first = dataset[0]
    C, T = int(first["image"].shape[0]), int(first["image"].shape[1])

    agg = np.zeros((C, T), dtype=np.int64)

    for i in tqdm(range(len(dataset)), desc=f"NaN stats ({split_name})", unit="img"):
        sample = dataset[i]
        img = sample["image"].detach().cpu().numpy()  # (C,T,H,W)
        counts = np.isnan(img).sum(axis=(2, 3)).astype(np.int64)  # (C,T)
        agg += counts

        # Per-sample outputs (optional)
        if save_per_sample:
            uid = sample.get("id", str(i))
            uid_safe = str(uid).replace("/", "_")
            _save_nan_counts(out_dir, f"{split_name}_{uid_safe}", counts)

    _save_nan_counts(out_dir, f"{split_name}_AGGREGATE", agg)