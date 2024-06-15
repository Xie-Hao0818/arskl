import math
import torch

import torch.nn as nn
from einops import rearrange
from pyskl.datasets.pipelines import PIPELINES
from arskl.transform.cutout import Cutout
import random


@PIPELINES.register_module()
class Vedio2Img(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, results):
        t1 = math.sqrt(results['imgs'].shape[2])
        results['imgs'] = rearrange(results['imgs'], 's c (t1 t2) h w-> s c (t1 h) (t2 w)', t1=int(t1)).contiguous()
        return results

@PIPELINES.register_module()
class CutoutVedio(nn.Module):
    def __init__(self, n_holes, length):
        super().__init__()
        self.cutout = Cutout(n_holes, length)

    def forward(self, results):
        s = results['imgs'].shape[0]
        for i in range(s):
            results['imgs'][i] = self.cutout(results['imgs'][i])
        return results


@PIPELINES.register_module()
class FrameCut(nn.Module):
    '''
    in: s c t h w
    '''

    def __init__(self, cut_ratio):
        super().__init__()
        self.cut_ratio = cut_ratio

    def forward(self, results):
        s, c, t, h, w = results['imgs'].shape
        black_img = torch.zeros((h, w))
        n_cut = self.cut_ratio * t
        cut_index = data = sorted(random.sample([i for i in range(t)], int(n_cut)))
        for i in range(s):
            vedio = results['imgs'][i]
            for t in range(vedio.shape[1]):
                if t in cut_index:
                    for j in range(vedio.shape[0]):
                        vedio[j][t] = black_img
        return results
