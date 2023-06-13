from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomRotation, ToTensor, Normalize, RandomCrop
from arskl.data.builder import DATASETS


@DATASETS.register_module()
class MNISTDataModule(LightningDataModule):
    def __init__(self, root, **kwargs):
        super().__init__()
        self.cfg = kwargs
        self.root = root
        self.train_transform = Compose([
            RandomCrop(28, padding=4),
            RandomRotation((-15, 15)),
            ToTensor(),
            Normalize((0.1306605041027069,), (0.308107852935791,)),
        ])
        self.valid_transform = Compose([
            ToTensor(),
            Normalize((0.1325145959854126,), (0.3104802370071411,))
        ])
        self.train = MNIST(root=self.root, train=True, transform=self.train_transform, download=True)
        self.val = MNIST(root=self.root, train=False, transform=self.valid_transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, pin_memory=True, **self.cfg)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False, pin_memory=True, **self.cfg)
