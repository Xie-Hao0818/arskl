from pyskl.models.cnns.x3d import X3D
from ..registry import MODEL
import torch.nn as nn
from pyskl.models.heads.simple_head import I3DHead
from einops import rearrange, reduce
from pyskl.models.cnns.c3d import C3D


@MODEL.register_module()
class X3D_MY(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = X3D(gamma_d=1,
                            in_channels=17,
                            base_channels=24,
                            num_stages=3,
                            se_ratio=None,
                            use_swish=False,
                            stage_blocks=(2, 5, 3),
                            spatial_strides=(2, 2, 2))
        self.head = I3DHead(num_classes=60, in_channels=216, dropout=0.5)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        out = self.backbone(x)
        out = self.head(out)
        out = reduce(out, '(b s) c->b c', 'mean', s=s)
        return out

    def init_weights(self):
        pass


@MODEL.register_module()
class C3D_MY(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = C3D(in_channels=17,
                            base_channels=32,
                            num_stages=3,
                            temporal_downsample=False)
        self.head = I3DHead(in_channels=256,
                            num_classes=60,
                            dropout=0.5)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        out = self.backbone(x)
        out = self.head(out)
        out = reduce(out, '(b s) c->b c', 'mean', s=s)
        return out

    def init_weights(self):
        pass
