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


#Using terratorch's MultiTemporalCropClassification dataset, will change this later
dataset_path = "multi-temporal-crop-classification-subset"

#Create a get datamodule function 

#Using terratorch's MultiTemporalCropClassificationDataModule, will change this later
datamodule = MultiTemporalCropClassificationDataModule(
    batch_size=8,
    num_workers=2,
    data_root=dataset_path,
    train_transform=[
        terratorch.datasets.transforms.FlattenTemporalIntoChannels(),  # Required for temporal data
        albumentations.D4(), # Random flips and rotation
        albumentations.pytorch.transforms.ToTensorV2(),
        terratorch.datasets.transforms.UnflattenTemporalFromChannels(n_timesteps=3),
    ],
    val_transform=None,  # Using ToTensor() by default
    test_transform=None,
    expand_temporal_dimension=True,
    use_metadata=False, # The crop dataset has metadata for location and time
    reduce_zero_label=True,
)




#function to flatten tensors to a more "tabular" format
def flatten_dataset(dataset, ignore_index=-1):
   
    X_list = []
    y_list = []

    for sample in tqdm(dataset, desc="Flattening dataset"):
        image = sample['image']  # shape: (C, T, H, W)
        mask = sample['mask']    # shape: (H, W)

        C, T, H, W = image.shape
        image = image.permute(2, 3, 1, 0)  # (H, W, T, C)
        image = image.reshape(H * W, T * C)  # (pixels, features)
        mask = mask.reshape(H * W)

        valid = mask != ignore_index
        X_list.append(image[valid].numpy())
        y_list.append(mask[valid].numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y

