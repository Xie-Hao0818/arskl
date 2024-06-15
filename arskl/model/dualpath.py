import torch
import torch.nn as nn
from einops import rearrange, reduce
from mmengine.model.weight_init import constant_init, kaiming_init

from .builder import build_model
from .stm import ConvBlock
from ..registry import MODEL


@MODEL.register_module()
class DualPath(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.body = build_model(
            dict(
                type='ResNet3d',
                backbone_cfg=dict(
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
                ),
                cls_head_cfg=dict(
                    in_channels=512,
                    num_classes=60,
                    dropout=0.5
                ),
                ckpt_path='/root/autodl-tmp/arskl/ckpt/ntu60_xsub/joint.pth'
            )
        )
        self.body.init_weights()
        self.body = self.body.model.backbone
        self.patch = nn.Sequential(
            nn.Conv3d(17, 32, (1, 3, 3), (1, 2, 2), (0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.Mish(inplace=True),
            nn.Conv3d(32, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.Mish(inplace=True),
            nn.Conv3d(32, 32, (1, 3, 3), (1, 1, 1), (0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.Mish(inplace=True),
            nn.AvgPool3d((1, 2, 2), (1, 2, 2)),

            ConvBlock(32, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            ConvBlock(64, 64, (3, 1, 1), (1, 1, 1), (1, 0, 0)),

            ConvBlock(64, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            ConvBlock(64, 64, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            nn.AvgPool3d((1, 2, 2), (1, 2, 2)),

            ConvBlock(64, 128, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            ConvBlock(128, 128, (3, 1, 1), (1, 1, 1), (1, 0, 0)),

            ConvBlock(128, 128, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            ConvBlock(128, 128, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            nn.AvgPool3d((1, 2, 2), (1, 2, 2)),

            ConvBlock(128, 256, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            ConvBlock(256, 256, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            nn.AvgPool3d((2, 1, 1), (2, 1, 1)),

            ConvBlock(256, 256, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            ConvBlock(256, 256, (3, 1, 1), (1, 1, 1), (1, 0, 0)),
            nn.AvgPool3d((2, 1, 1), (2, 1, 1)),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 60),
        )

    def init_weights(self):
        for m in self.patch.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        with torch.no_grad():
            out1 = self.pool(self.body(x))
        out2 = self.pool(self.patch(x))
        out = torch.cat([out1, out2], 1)
        out = self.head(out)
        out = reduce(out, '(b s) c ->b c', 'mean', s=s)
        return out
