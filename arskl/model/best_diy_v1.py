import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange
from einops.layers.torch import Reduce, Rearrange
from mmengine.model.weight_init import constant_init, kaiming_init

from ..registry import MODEL


class SpaceMult(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 ):
        super().__init__()
        self.HW = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            (kernel_size - 1) // 2,
                            groups=groups,
                            bias=False,
                            )
        self.t_pool = nn.AvgPool3d((stride, 1, 1), (stride, 1, 1), ceil_mode=True)

    def forward(self, x):
        t = x.shape[2]
        x = rearrange(x, 'b c t h w->(b t) c h w')
        out = self.HW(x)
        out = rearrange(out, '(b t) c h w->b c t h w', t=t)
        out = self.t_pool(out)
        return out


class TimeMult(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 groups=1,
                 ):
        super().__init__()
        self.C = nn.Conv3d(in_channels,
                           out_channels,
                           (kernel_size, 1, 1),
                           (stride, 1, 1),
                           ((kernel_size - 1) // 2, 0, 0),
                           groups=groups,
                           bias=False)
        self.s_pool = nn.AvgPool3d((1, stride, stride), (1, stride, stride), ceil_mode=True)

    def forward(self, x):
        y = self.C(x)
        out = self.s_pool(y)
        return out


class M3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 ):
        super().__init__()
        self.CS = SpaceMult(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            groups,
                            )
        self.CT = TimeMult(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           groups,
                           )

    def forward(self, x):
        return self.CS(x) + self.CT(x)


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ta = nn.Sequential(
            Reduce('b c t h w->b 1 t', 'mean'),
            nn.Conv1d(1, 1, 1, 1, 0, bias=False),
            Rearrange('b 1 t->b t 1 1 1'),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        x1 = rearrange(x, 'b c t h w->b t c h w')
        out = torch.multiply(x1, self.ta(x))
        out = rearrange(out, 'b t c h w->b c t h w')
        return out


class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1, is_use=True):
        super().__init__()
        t = int(abs((b + math.log(channel, 2)) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Hardsigmoid()
        self.is_use = is_use

    def forward(self, x):
        if not self.is_use:
            return x
        y = reduce(x, 'b c t h w->b 1 c', 'mean')
        y = self.conv(y)
        y = rearrange(y, 'b 1 c->b c 1 1 1')
        y = self.sigmoid(y)
        return torch.multiply(x, y)


class TSM(nn.Module):
    def __init__(self, is_use=True):
        super().__init__()
        self.is_use = is_use

    def forward(self, x):
        if not self.is_use:
            return x
        _, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w->b t c (h w)')
        left_split, mid_split, right_split = x.chunk(3, dim=2)
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)
        out = torch.cat((left_split, mid_split, right_split), 2)
        out = rearrange(out, 'b t c (h w)->b c t h w', h=h, w=w)
        return out


class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g c) t h w->b (c g) t h w', g=self.groups)
        return x


class BRC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dowansample=False,
                 active=True):
        super().__init__()
        self.P = nn.AvgPool3d(stride, stride, ceil_mode=True)
        self.A = nn.ReLU()
        self.C = M3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if dowansample else stride,
            groups=groups,
        )
        self.N = nn.BatchNorm3d(in_channels)
        self.dowansample = dowansample
        self.active = active

    def forward(self, x):
        out = self.N(x)
        if (not self.dowansample) and self.active:
            out = self.A(out)
        out = self.C(out)
        if self.dowansample:
            out = self.P(out)
        return out


class Bottleneck3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            drop_path=0.2,
            drop_res=0.5,
            bottle_ratio=0.25,
            use_tsm=True,
            use_eca=True,
    ):
        super().__init__()
        planes = int(in_channels * bottle_ratio)
        self.drop_path = torch.bernoulli(torch.tensor([1]), p=drop_path) if self.training else 0
        self.drop_res = torch.bernoulli(torch.tensor([1]), p=drop_res) if self.training else 0
        self.active = nn.ReLU()
        self.downsample = BRC(in_channels, out_channels, stride=stride, dowansample=True)
        self.neck = nn.Sequential(
            TSM(is_use=use_tsm),
            BRC(in_channels, planes),
            BRC(planes, planes, stride=stride, kernel_size=3),
            ECA(planes, is_use=use_eca),
            BRC(planes, out_channels)
        )

    def forward(self, x):
        if self.drop_path:
            return self.downsample(x)
        if self.drop_res:
            return self.neck(x)
        return self.downsample(x) + self.neck(x)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


@MODEL.register_module()
class ResNet3D_Diy_Best_v1(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel = 17
        num_classes = 120
        channel = [64, 128, 256, 512]
        drop_ratio = 0.5
        self.stem = nn.Sequential(
            nn.BatchNorm3d(in_channel),
            nn.ReLU(),
            nn.Conv3d(in_channel, channel[0], (1, 3, 3), (1, 2, 2), (0, 1, 1)),  # 48,28,28
            nn.AvgPool3d(3, 2, 1)  # 24,14,14
        )
        self.stage1 = nn.Sequential(
            Bottleneck3d(channel[0], channel[1]),
            Bottleneck3d(channel[1], channel[1]),
            Bottleneck3d(channel[1], channel[1]),
            Bottleneck3d(channel[1], channel[1], 2),  # 12,7,7
            # TemporalAttention()
        )
        self.stage2 = nn.Sequential(
            Bottleneck3d(channel[1], channel[2]),
            Bottleneck3d(channel[2], channel[2]),
            Bottleneck3d(channel[2], channel[2]),
            Bottleneck3d(channel[2], channel[2]),
            Bottleneck3d(channel[2], channel[2]),
            Bottleneck3d(channel[2], channel[2], 2),  # 6,4,4
            # TemporalAttention()
        )
        self.stage3 = nn.Sequential(
            Bottleneck3d(channel[2], channel[3]),
            Bottleneck3d(channel[3], channel[3]),
            Bottleneck3d(channel[3], channel[3], 2),  # 3,2,2
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(drop_ratio),
            nn.Flatten(),
            nn.Linear(channel[3], num_classes)
        )

    def forward(self, x):
        # s = x.shape[1]
        # x = rearrange(x, 'b s c t h w->(b s) c t h w')
        out1 = self.stem(x)
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        out = self.stage3(out3)
        # out = self.head(out)
        # out = reduce(out, '(b s) c->b c', 'mean', s=s)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                constant_init(m, 1)
