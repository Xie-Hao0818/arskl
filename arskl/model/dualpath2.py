import torch
import torch.nn as nn
from einops import rearrange, reduce

from .builder import build_model
from ..registry import MODEL


@MODEL.register_module()
class DualPath2(nn.Module):
    def __init__(self, ckpt_path, num_classes):
        super().__init__()
        self.body1 = build_model(
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
                    num_classes=num_classes,
                    dropout=0.5
                ),
                ckpt_path=ckpt_path
            )
        )
        self.body2 = build_model(
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
                    num_classes=num_classes,
                    dropout=0.5
                ),
                ckpt_path=ckpt_path
            )
        )
        self.body1.init_weights()
        self.body2.init_weights()
        self.path1 = self.body1.model.backbone
        self.path2 = self.body2.model.backbone
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def init_weights(self):
        # for m in self.path2.modules():
        #     if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        #         kaiming_init(m)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         constant_init(m, 1)
        pass

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        with torch.no_grad():
            out1 = self.pool(self.path1(x))
        out2 = self.pool(self.path2(x))
        out = torch.cat([out1, out2], 1)
        out = self.head(out)
        out = reduce(out, '(b s) c ->b c', 'mean', s=s)
        return out
