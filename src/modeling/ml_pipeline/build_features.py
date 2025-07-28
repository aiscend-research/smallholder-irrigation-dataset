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
def get_datamodule(dataset_path: str, batch_size: int = 8, num_workers: int = 2, n_timesteps: int = 3):
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
    return datamodule


#function to flatten tensors to a more "tabular" format
def flatten_dataset(dataset, ignore_index=-1):
    """
    Flattens multi-temporal dataset for sklearn models.
    Supports both single-band and multi-band masks.

    Returns:
        X: np.ndarray of shape (N, T*C)
        y: np.ndarray of shape (N,) for single-band or (N, B) for multi-band
    """
    X_list = []
    y_list = []

    for sample in tqdm(dataset, desc="Flattening dataset"):
        image = sample['image']  # shape: (C, T, H, W)
        mask = sample['mask']    # shape: (H, W) or (B, H, W)

        C, T, H, W = image.shape
        image = image.permute(2, 3, 1, 0)  # (H, W, T, C)
        image = image.reshape(H * W, T * C)

        # Handle mask flattening
        if mask.ndim == 2:
            mask_flat = mask.reshape(H * W)
            valid = mask_flat != ignore_index
            y_valid = mask_flat[valid].numpy()
        elif mask.ndim == 3:
            B = mask.shape[0]
            mask_flat = mask.permute(1, 2, 0).reshape(H * W, B)  # (H*W, B)
            valid = ~torch.any(mask_flat == ignore_index, dim=1)  # Exclude rows where *any* band is ignore_index
            y_valid = mask_flat[valid].numpy()
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        X_valid = image[valid].numpy()

        X_list.append(X_valid)
        y_list.append(y_valid)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y
