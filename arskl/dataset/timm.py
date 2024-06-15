from lightning import LightningDataModule
from timm.data import create_dataset
from torch.utils.data import DataLoader

from ..registry import DATASETS
from ..transform.builder import build_transform


@DATASETS.register_module()
class TimmDataModule(LightningDataModule):
    def __init__(self, train_transform_cfg, valid_transform_cfg, train_dataset_cfg, valid_dataset_cfg,
                 train_dataloader_cfg, valid_dataloader_cfg):
        super().__init__()
        self.train = create_dataset(transform=build_transform(train_transform_cfg), **train_dataset_cfg)
        self.val = create_dataset(transform=build_transform(valid_transform_cfg), **valid_dataset_cfg)
        self.train_dataloader_cfg = train_dataloader_cfg
        self.valid_dataloader_cfg = valid_dataloader_cfg

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.val, **self.valid_dataloader_cfg)

    def test_dataloader(self):
        return DataLoader(self.val, **self.valid_dataloader_cfg)
