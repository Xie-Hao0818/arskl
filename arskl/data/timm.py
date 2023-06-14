from lightning import LightningDataModule
from timm.data import create_dataset
from torch.utils.data import DataLoader

from arskl.data.builder import DATASETS
from arskl.transform.builder import build_transform


@DATASETS.register_module()
class TimmDataModule(LightningDataModule):
    def __init__(self, traintf_cfg, valtf_cfg, trainds_cfg, valds_cfg, traindl_cfg, valdl_cfg):
        super().__init__()
        self.train = create_dataset(transform=build_transform(traintf_cfg), **trainds_cfg)
        self.val = create_dataset(transform=build_transform(valtf_cfg), **valds_cfg)
        self.traindl_cfg = traindl_cfg
        self.valdl_cfg = valdl_cfg

    def train_dataloader(self):
        return DataLoader(self.train, **self.traindl_cfg)

    def val_dataloader(self):
        return DataLoader(self.val, **self.valdl_cfg)
