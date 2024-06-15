import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange

from ..registry import TRANSFORM
from .pose import to_tensor


@TRANSFORM.register_module()
class PoseToTensor(nn.Module):
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self):
        super(PoseToTensor, self).__init__()

    def forward(self, x):
        """Performs the ToTensor formatting.

        Args:
            x (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = []
        for item in x:
            data.append(to_tensor(item))
        return data


@TRANSFORM.register_module()
class PoseGCNInput(nn.Module):
    """
    x[
        data: c v m t
        label : 1
    ]
    """

    def __init__(self, patten='c v m t'):
        super(PoseGCNInput, self).__init__()
        self.patten = patten

    def forward(self, x):
        keypoint = x['keypoint']
        keypoint_score = x['keypoint_score']
        data, _ = pack([keypoint, keypoint_score], 'n t v *')
        data = rearrange(data, 'm t v c->' + self.patten)
        label = np.array((x['label']))
        x = [
            data,
            label,
        ]
        return x


@TRANSFORM.register_module()
class ManPad(nn.Module):
    """

    """

    def __init__(self, max_man=26, patten='c t v m'):
        super(ManPad, self).__init__()
        self.max_man = max_man
        self.patten = patten

    def forward(self, x):
        data, label = x
        pad_shape = list(data.shape)
        pad_shape[2] = self.max_man - data.shape[2]
        pad = torch.zeros(pad_shape)
        data = torch.cat([pad, torch.tensor(data)], 2)
        data = rearrange(data, 'c v m t->' + self.patten)
        x = [
            data,
            label,
        ]
        return x
