from lightning import LightningDataModule
from pyskl.datasets.dataset_wrappers import RepeatDataset
from pyskl.datasets.pose_dataset import PoseDataset
from torch.utils.data import DataLoader

from ..registry import DATASETS


@DATASETS.register_module()
class PoseDataLoaderX(LightningDataModule):
    def __init__(self, train_dataset_cfg, valid_dataset_cfg, test_dataset_cfg,
                 train_dataloader_cfg, valid_dataloader_cfg, test_dataloader_cfg):
        super().__init__()
        self.train = RepeatDataset(**train_dataset_cfg)
        self.valid = PoseDataset(**valid_dataset_cfg)
        self.test = PoseDataset(**test_dataset_cfg)
        self.train_dataloader_cfg = train_dataloader_cfg
        self.valid_dataloader_cfg = valid_dataloader_cfg
        self.test_dataloader_cfg = test_dataloader_cfg

    def train_dataloader(self):
        return DataLoader(self.train, **self.train_dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.valid, **self.valid_dataloader_cfg)

    def test_dataloader(self):
        return DataLoader(self.test, **self.test_dataloader_cfg)
