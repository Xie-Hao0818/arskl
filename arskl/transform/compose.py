from collections.abc import Sequence

import torch.nn as nn

from ..registry import TRANSFORM, build_from_cfg


@TRANSFORM.register_module()
class Compose(nn.Module):
    """Compose a dataset pipeline with a sequence of transforms.

        Args:
            transforms (list[dict | callable]):
                Either config dicts of transforms or transform objects.
        """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, TRANSFORM)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def forward(self, x):
        """Call function to apply transforms sequentially.

        Args:
            x (dict): A result dict contains the dataset to transform.

        Returns:
            dict: Transformed dataset.
        """
        for t in self.transforms:
            x = t(x)
        return x
