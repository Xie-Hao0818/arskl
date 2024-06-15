import torch
import torch.nn as nn
from einops import rearrange, reduce
from mmengine.model.weight_init import constant_init, kaiming_init

from .builder import build_model
from ..registry import MODEL


# 假设快路径的特征通道数是慢路径的2倍
class LateralConnection(nn.Module):
    def __init__(self, fast_channels, slow_channels):
        super(LateralConnection, self).__init__()
        # 使用1x1卷积调整快路径特征的通道数
        self.conv1x1 = nn.Conv2d(fast_channels, slow_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, slow_feature, fast_feature):
        # 调整快路径特征图的通道数以匹配慢路径
        fast_feature_adjusted = self.conv1x1(fast_feature)
        # 直接相加作为一种融合策略
        combined_feature = slow_feature + fast_feature_adjusted
        return combined_feature

@MODEL.register_module()
class DualPath4(nn.Module):
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
                    dropout=0.5   # 0.5
                ),
                ckpt_path=ckpt_path
            )
        )
        self.body2 = build_model(
            dict(
                type='ResNet3d',
                backbone_cfg=dict(
                    in_channels=17,
                    base_channels=32,  # 32
                    num_stages=3,  # 3
                    out_indices=(2,),
                    stage_blocks=(1, 1, 1),  # (1, 1, 1)
                    conv1_stride=(1, 1),  # (1, 1)
                    pool1_stride=(1, 1),  # (1, 1)
                    inflate=(0, 1, 1),  # (0, 1, 1)
                    spatial_strides=(2, 2, 2),  # (2, 2, 2)
                    temporal_strides=(2, 2, 2)  # (2, 2, 2)
                ),
                cls_head_cfg=dict(
                    in_channels=512,
                    num_classes=400,
                    dropout=0.5
                ),
                # ckpt_path='/root/autodl-tmp/ckpt/Kinetics_400/joint.pth'
                ckpt_path=None
            )
        )
        self.body3 = build_model(
            dict(
                type='ResNet3D_Diy_Best_v1'
            )
        )

        self.body1.init_weights()
        self.body2.init_weights()
        self.path1 = self.body1.model.backbone
        self.path2 = self.body2.model.backbone
        self.path3 = self.body3
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                constant_init(m, 1)
        self.body1.init_weights()
        pass

    def forward(self, x):
        b, s, c, t, h, w = x.shape
        x = rearrange(x, 'b s c t h w->(b s) c t h w')

        # 双路径================================================================
        with torch.no_grad():
            out1 = self.pool(self.path1(x))
        out2 = self.pool(self.path3(x))
        out = torch.cat([out1, out2], 1)
        out = self.head(out)
        # 双路径================================================================

        # 基准路径================================================================
        # out = self.body1.model(x)
        # 基准路径================================================================

        out = reduce(out, '(b s) c ->b c', 'mean', s=s)
        return out


@MODEL.register_module()
class DualPath5(nn.Module):
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
                ckpt_path='/root/autodl-tmp/ckpt/ntu60_xsub/joint.pth'
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
        #     elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
        #         constant_init(m, 1)
        pass

    def forward(self, x):
        s = x.shape[2]
        x = rearrange(x, 'g b s c t h w->g (b s) c t h w')

        # 双路径================================================================
        with torch.no_grad():
            out1 = self.pool(self.path1(x[0]))
            out2 = self.pool(self.path2(x[1]))
        out = torch.cat([out1, out2], 1)
        out = self.head(out)
        # 双路径================================================================

        # 基准路径================================================================
        # out = self.body1.model(x)
        # 基准路径================================================================

        out = reduce(out, '(b s) c ->b c', 'mean', s=s)
        return out
