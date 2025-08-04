import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from custom_dataset import MultiTemporalCropDataset


class MultiTemporalCropDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_files,
        val_files=None,
        test_files=None,
        batch_size=4,
        num_workers=0,
        label_bands=list(range(1, 9))
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_bands = label_bands
        
        # Data directory contains both .tif and .json files
        self.data_dir = data_dir

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = MultiTemporalCropDataset(
                self.data_dir, self.train_files, self.label_bands
            )
        if stage in (None, "fit", "validate"):
            if self.val_files:
                self.val_dataset = MultiTemporalCropDataset(
                    self.data_dir, self.val_files, self.label_bands
                )
        if stage in (None, "test"):
            if self.test_files:
                self.test_dataset = MultiTemporalCropDataset(
                    self.data_dir, self.test_files, self.label_bands
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return None

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return None 