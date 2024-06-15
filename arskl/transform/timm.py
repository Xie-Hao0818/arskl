from timm.data import create_transform
from torchvision.transforms.transforms import Compose

from ..registry import TRANSFORM


@TRANSFORM.register_module()
class TimmTransform(Compose):
    def __init__(self, **transform_cfg):
        super().__init__(transforms=create_transform(**transform_cfg).transforms)
