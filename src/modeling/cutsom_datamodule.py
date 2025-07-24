from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class MultiTemporalBarebonesDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_files,
        val_files=None,
        test_files=None,
        batch_size=4,
        num_workers=0,
        image_band=1,
        mask_band=2
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_band = image_band
        self.mask_band = mask_band

    def setup(self, stage=None):
        from custom_dataset import MultiTemporalBarebonesDataset
        if stage in (None, "fit"):
            self.train_dataset = MultiTemporalBarebonesDataset(
                self.data_dir, self.train_files, self.image_band, self.mask_band
            )
        if stage in (None, "fit", "validate"):
            if self.val_files:
                self.val_dataset = MultiTemporalBarebonesDataset(
                    self.data_dir, self.val_files, self.image_band, self.mask_band
                )
        if stage in (None, "test"):
            if self.test_files:
                self.test_dataset = MultiTemporalBarebonesDataset(
                    self.data_dir, self.test_files, self.image_band, self.mask_band
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