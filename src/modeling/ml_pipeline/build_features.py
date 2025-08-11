import numpy as np
from tqdm import tqdm


def flatten_dataset(dataset, ignore_value_in_image=None):
    """
    Flattens a multi-temporal crop dataset for ML.
    Returns all image features and all mask bands (single/multi).

    Args:
        dataset: PyTorch Dataset where each sample is a dict:
            'image': Tensor (C, T, H, W)
            'mask' : Tensor (H, W) or (B, H, W)
        ignore_value_in_image: Optional value in image pixels to ignore (e.g., -9999 for clouds)

    Returns:
        X: np.ndarray, shape (N, C*T)
        y: np.ndarray, shape (N,) or (N, B)

    Notes:
        Pixels where either the mask or image contains NaN in any band are filtered out.
    """
    X_list = []
    y_list = []
    
    total_pixels = 0
    valid_pixels = 0

    for i, sample in enumerate(tqdm(dataset, desc="Flattening dataset")):
        image = sample['image']  # (C, T, H, W)
        mask = sample['mask']    # (H, W) or (B, H, W)

        C, T, H, W = image.shape
        image = image.permute(2, 3, 1, 0)  # (H, W, T, C)
        image_flat = image.reshape(H * W, T * C)

        # Handle mask: always return all bands
        if mask.ndim == 2:
            mask_flat = mask.reshape(H * W, 1)  # (N, 1) for consistency
        elif mask.ndim == 3:
            B = mask.shape[0]
            mask_flat = mask.permute(1, 2, 0).reshape(H * W, B)  # (N, B)
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # Convert to numpy for easier handling
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