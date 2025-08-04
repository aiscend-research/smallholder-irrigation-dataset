import os
import sys
import torch
import gdown
import terratorch
import albumentations
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
import warnings
import numpy as np
from tqdm import tqdm


#get datamodule function 
def get_datamodule(dataset_path: str, batch_size: int = 8, num_workers: int = 2, n_timesteps: int = 3, 
                   datamodule_type: str = "terratorch", **kwargs):
    """
    Get datamodule based on specified type.
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        n_timesteps: Number of timesteps (for terratorch)
        datamodule_type: Either "terratorch" or "custom"
        **kwargs: Additional arguments for custom datamodule
    """
    if datamodule_type.lower() == "terratorch":
        datamodule = MultiTemporalCropClassificationDataModule(
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=dataset_path,
            train_transform=[
                terratorch.datasets.transforms.FlattenTemporalIntoChannels(),  # Required for temporal data
                albumentations.D4(), # Random flips and rotation
                albumentations.pytorch.transforms.ToTensorV2(),
                terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=3),
            ],
            val_transform=None,
            test_transform=None,
            expand_temporal_dimension=True,
            use_metadata=False,
            reduce_zero_label=True,
        )
    elif datamodule_type.lower() == "custom":
        # Import here to avoid circular imports
        from custom_datamodule import MultiTemporalCropDataModule
        
        # Extract custom datamodule specific parameters
        train_files = kwargs.get('train_files', [])
        val_files = kwargs.get('val_files', None)
        test_files = kwargs.get('test_files', None)
        label_bands = kwargs.get('label_bands', list(range(1, 9)))
        
        datamodule = MultiTemporalCropDataModule(
            data_dir=dataset_path,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            batch_size=batch_size,
            num_workers=num_workers,
            label_bands=label_bands
        )
    else:
        raise ValueError(f"Unknown datamodule_type: {datamodule_type}. Must be 'terratorch' or 'custom'")
    
    return datamodule


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
