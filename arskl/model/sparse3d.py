import torch
from einops import rearrange, reduce
from mmengine.model.weight_init import constant_init, kaiming_init
from spconv.pytorch import SparseConvTensor, SparseSequential, SparseConv3d, SparseModule, SubMConv3d, SparseMaxPool3d
from torch import nn

from ..registry import MODEL


# os.environ["SPCONV_BENCHMARK"] = "1"
# os.environ["SPCONV_DEBUG"] = "1"


def to_sparse(data):
    B, C, T, H, W = data.shape
    # 获取非零值的坐标
    non_zero_indices = torch.nonzero(data, as_tuple=False)
    # 构建 SparseConvTensor 所需的索引和值
    indices = non_zero_indices[:, [0, 2, 3, 4]].to(torch.int32)
    values = data[non_zero_indices[:, 0], :, non_zero_indices[:, 2], non_zero_indices[:, 3],
             non_zero_indices[:, 4]].view(-1, C)
    sparse_tensor = SparseConvTensor(features=values, indices=indices, spatial_shape=[T, H, W],
                                     batch_size=B)
    return sparse_tensor


class SparseBottleneck3d_S(SparseModule):
    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 bottle_ratio=0.25,
                 ):
        super().__init__()
        planes = int(inplanes * bottle_ratio)
        self.conv1 = SparseSequential(
            SubMConv3d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = SparseSequential(
            SparseConv3d(in_channels=planes, out_channels=planes,
                         kernel_size=(1, 3, 3),
                         stride=(1, stride, stride),
                         padding=(0, 1, 1),
                         bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = SparseSequential(
            SubMConv3d(in_channels=planes, out_channels=outplanes, kernel_size=1, bias=False),
            nn.BatchNorm1d(outplanes),
        )
        self.relu = SparseSequential(
            nn.ReLU(inplace=True),
        )
        self.downsample = SparseSequential(
            SubMConv3d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, bias=False),
            nn.BatchNorm1d(outplanes),
            SparseMaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, 1, 1,)
            )
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out


class SparseBottleneck3d_S_T(SparseModule):
    def __init__(self,
                 inplanes,
                 outplanes,
                 s_stride=1,
                 t_stride=1,
                 bottle_ratio=0.25,
                 ):
        super().__init__()
        planes = int(inplanes * bottle_ratio)
        self.conv1 = SparseSequential(
            SparseConv3d(in_channels=inplanes, out_channels=planes,
                         kernel_size=(3, 1, 1),
                         stride=(t_stride, 1, 1),
                         padding=(1, 0, 0),
                         bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = SparseSequential(
            SparseConv3d(in_channels=planes, out_channels=planes,
                         kernel_size=(1, 3, 3),
                         stride=(1, s_stride, s_stride),
                         padding=(0, 1, 1),
                         bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = SparseSequential(
            SubMConv3d(in_channels=planes, out_channels=outplanes, kernel_size=1, bias=False),
            nn.BatchNorm1d(outplanes),
        )
        self.relu = SparseSequential(
            nn.ReLU(inplace=True),
        )
        self.downsample = SparseSequential(
            SubMConv3d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, bias=False),
            nn.BatchNorm1d(outplanes),
            SparseMaxPool3d(
                kernel_size=(3, 3, 3),
                stride=(t_stride, s_stride, s_stride),
                padding=(1, 1, 1,)
            )
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out


@MODEL.register_module()
class SparseResnet(SparseModule):
    def __init__(self):
        super().__init__()
        channel = [32, 128, 256, 512]
        self.stem = SparseSequential(
            SparseConv3d(in_channels=17, out_channels=channel[0], kernel_size=(1, 7, 7), stride=(1, 1, 1),
                         padding=(0, 3, 3), bias=False),
            nn.BatchNorm1d(channel[0]),
            nn.ReLU(inplace=True),
            SparseMaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )
        self.net = SparseSequential(
            SparseBottleneck3d_S(channel[0], channel[1], stride=2),
            SparseBottleneck3d_S(channel[1], channel[1], stride=1),

            SparseBottleneck3d_S_T(channel[1], channel[2], s_stride=2, t_stride=2),
            SparseBottleneck3d_S_T(channel[2], channel[2], s_stride=1, t_stride=1),
            SparseBottleneck3d_S_T(channel[2], channel[2], s_stride=1, t_stride=1),
            SparseBottleneck3d_S_T(channel[2], channel[2], s_stride=1, t_stride=1),

            SparseBottleneck3d_S_T(channel[2], channel[3], s_stride=2, t_stride=2),
            SparseBottleneck3d_S_T(channel[3], channel[3], s_stride=1, t_stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(channel[3], 60)
        )

    def forward(self, x):
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')  # torch.Size([4, 17, 12, 64, 64])
        x = to_sparse(x)
        out = self.stem(x)
        out = self.net(out)
        out = self.head(out.dense())
        out = reduce(out, '(b s) c->b c', 'mean', s=s)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d, SubMConv3d, SparseConv3d)):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                constant_init(m, 1)
            if isinstance(m, (SparseBottleneck3d_S, SparseBottleneck3d_S_T)):
                constant_init(m.conv3[1], 0)
