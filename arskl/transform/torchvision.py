from torchvision.transforms.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize, \
    RandomErasing, CenterCrop

from ..registry import TRANSFORM


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


@TRANSFORM.register_module()
class Resize(Resize):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@TRANSFORM.register_module()
class RandomErasing(RandomErasing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@TRANSFORM.register_module()
class CenterCrop(CenterCrop):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
