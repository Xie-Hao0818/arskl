import torch.nn as nn
from timm.data.auto_augment import augment_and_mix_transform
from timm.data.auto_augment import auto_augment_transform
from timm.data.auto_augment import rand_augment_transform

from ..registry import TRANSFORM


@TRANSFORM.register_module()
class MixAugment(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        aat = augment_and_mix_transform(**self.kwargs)
        return aat(x)


@TRANSFORM.register_module()
class AutoAugment(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        aat = auto_augment_transform(**self.kwargs)
        return aat(x)


@TRANSFORM.register_module()
class RandAugment(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        rat = rand_augment_transform(**self.kwargs)
        return rat(x)
