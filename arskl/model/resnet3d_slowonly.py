import torch.nn as nn
from einops import rearrange, reduce
from pyskl.models.heads.simple_head import I3DHead

from .resnet3d import ResNet3dSlowOnly_M
from ..registry import MODEL


@MODEL.register_module()
class ResNet3d_Skl(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = ResNet3dSlowOnly_M(**backbone)
        self.head = I3DHead(**head)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        feature = self.backbone(x)
        soft = self.head(feature)
        soft = reduce(soft, '(b s) c ->b c', 'mean', s=s)
        return soft

    def init_weights(self):
        pass


@MODEL.register_module()
class ResNet3d_Skl_NTU60(ResNet3d_Skl):
    def __init__(self):
        backbone = dict(
            in_channels=17,
            base_channels=32,
            num_stages=3,
            out_indices=(2,),
            stage_blocks=(4, 6, 3),
            conv1_stride=(1, 1),
            pool1_stride=(1, 1),
            inflate=(0, 1, 1),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 2)
        )
        head = dict(
            in_channels=512,
            num_classes=60,
            dropout=0.5
        )
        super().__init__(backbone, head)
