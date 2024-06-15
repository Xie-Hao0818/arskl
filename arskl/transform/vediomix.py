import torch
import torch.nn as nn
from einops import rearrange


class VedioMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.rand(1).cuda()

    def forward(self, x, y):
        x, y = x.cuda(), y.cuda()
        s = x.shape[1]
        x = rearrange(x, 'b s c t h w->(b s) c t h w')
        B, C, T, H, W = x.shape
        wc, wh = torch.randint(0, W, (1,)).cuda(), torch.randint(0, H, (1,)).cuda()
        w1, w2 = self.clip(wc - W * (0.5 * torch.sqrt(self.alpha)), W), self.clip(
            wc + W * (0.5 * torch.sqrt(self.alpha)), W)
        h1, h2 = self.clip(wh - H * (0.5 * torch.sqrt(self.alpha)), H), self.clip(
            wh + H * (0.5 * torch.sqrt(self.alpha)), H)
        t1, t2 = 0, T
        for i in range(B):
            i2 = int(torch.randint(0, B, (1,)))
            x[i, :, t1:t2, h1:h2, w1:w2] = x[i2, :, t1:t2, h1:h2, w1:w2]
            y[i] = self.alpha * y[i] + (1 - self.alpha) * y[i2]
        x = rearrange(x, '(b s) c t h w->b s c t h w', s=s)
        return x, y

    @staticmethod
    def clip(w, W):
        x = torch.max(torch.tensor((w, W))).cuda()
        y = torch.min(torch.tensor((x, 0))).cuda()
        return int(y)
