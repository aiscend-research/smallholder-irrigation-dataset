import numpy as np
from tqdm import tqdm


def get_datasets(data_dir: str, train_files: list, val_files: list = None, test_files: list = None,
                 label_bands: list = None):
    """
    Get custom datasets for irrigation classification.
    
    Args:
        data_dir: Path to the data directory containing .tif and .json files
        train_files: List of training sample filenames (without extension)
        val_files: List of validation sample filenames (without extension)
        test_files: List of test sample filenames (without extension)
        label_bands: List of label band indices to use (1-based, default: [1,2])
    
    Returns:
        dict: Contains 'train_dataset', 'val_dataset', 'test_dataset' (if provided)
    """
    # Import here to avoid circular imports
    from custom_dataset import MultiTemporalCropDataset
    
    # Set default label bands if not provided
    if label_bands is None:
        label_bands = [1, 2]
    
    datasets = {}
    
    # Create train dataset
    datasets['train_dataset'] = MultiTemporalCropDataset(
        data_dir=data_dir,
        sample_file_list=train_files,
        label_bands=label_bands
    )
    
    # Create validation dataset if provided
    if val_files:
        datasets['val_dataset'] = MultiTemporalCropDataset(
            data_dir=data_dir,
            sample_file_list=val_files,
            label_bands=label_bands
        )
    
    # Create test dataset if provided
    if test_files:
        datasets['test_dataset'] = MultiTemporalCropDataset(
            data_dir=data_dir,
            sample_file_list=test_files,
            label_bands=label_bands
        )
    
    return datasets


#function to flatten tensors to a more "tabular" format
def flatten_dataset(dataset, ignore_index=-1, ignore_value_in_image=None):
    """
    Flattens a multi-temporal crop dataset for ML.
    Returns all image features and all mask bands (single/multi).

    Args:
        dataset: PyTorch Dataset where each sample is a dict:
            'image': Tensor (C, T, H, W)
            'mask' : Tensor (H, W) or (B, H, W)
        ignore_index: Mask value to skip
        ignore_value_in_image: Optional value in image pixels to ignore (e.g., -9999 for clouds)

    Returns:
        X: np.ndarray, shape (N, C*T)
        y: np.ndarray, shape (N,) or (N, B)
    """
    X_list = []
    y_list = []

    for sample in tqdm(dataset, desc="Flattening dataset"):
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

        # Validity: ignore if *any* band is ignore_index in that pixel
        valid_mask = ~np.any(mask_flat.numpy() == ignore_index, axis=1)

        if ignore_value_in_image is not None:
            valid_image = ~np.any(image_flat.numpy() == ignore_value_in_image, axis=1)
            valid = valid_mask & valid_image
        else:
            valid = valid_mask

        X_valid = image_flat[valid].numpy()
        y_valid = mask_flat[valid].numpy()

        X_list.append(X_valid)
        y_list.append(y_valid)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # If only one mask band, squeeze to (N,)
    if y.shape[1] == 1:
        y = y.squeeze(1)

    return X, y