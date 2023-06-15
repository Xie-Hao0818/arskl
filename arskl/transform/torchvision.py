from torchvision.transforms.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

from .builder import TRANSFORM


@TRANSFORM.register_module()
class RandomCrop(RandomCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@TRANSFORM.register_module()
class RandomHorizontalFlip(RandomHorizontalFlip):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@TRANSFORM.register_module()
class ToTensor(ToTensor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@TRANSFORM.register_module()
class Normalize(Normalize):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)