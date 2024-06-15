import torch
import torch.nn as nn
from einops import reduce, rearrange

from pyskl.datasets.pipelines import PIPELINES


@PIPELINES.register_module()
class NozeroSample(nn.Module):
    def __init__(self):
        super(NozeroSample, self).__init__()

    def forward(self, results):
        # s, c, t, h, w = results['imgs'].shape
        info_metric = reduce(results['imgs'], 's c t h w->s', 'mean')
        index = torch.argmax(info_metric)
        results['imgs'] = results['imgs'][index]
        results['imgs'] = rearrange(results['imgs'], 'c t h w->1 c t h w')
        return results
