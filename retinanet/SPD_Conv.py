import torch
from torch import nn
import torch.nn.functional as F


def padding(ts, type):
    b, c, h, w = ts.shape
    if type == 1:
        # if w > h:
        #     ts = F.pad(ts, (0, 0, 0, w - h), 'constant', 0)
        # elif h > w:
        #     ts = F.pad(ts, (0, h - w, 0, 0), 'constant', 0)
        if w % 2 == 0 and h % 2 == 0:
            return ts
        elif w % 2 == 0 and h % 2 != 0:
            ts = F.pad(ts, (0, 0, 0, 1), 'constant', 0)
        elif w % 2 != 0 and h % 2 == 0:
            ts = F.pad(ts, (0, 1, 0, 0), 'constant', 0)
        else:
            ts = F.pad(ts, (0, 1, 0, 1), 'constant', 0)
    elif type == 2:
        if w > h:
            ts = F.pad(ts, (0, 0, 0, w - h), 'constant', 0)
        elif w < h:
            ts = F.pad(ts, (0, h - w, 0, 0), 'constant', 0)
        else:
            return ts
        if w % 2 != 0:
            ts = F.pad(ts, (0, 1, 0, 1), 'constant', 0)
    return ts


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # b, c, w, h = x.shape
        # if w % 2 == 0 and h % 2 != 0:
        #     x = F.pad(x, (0, 0, 0, 1), 'constant', 0)
        # elif w % 2 != 0 and h % 2 == 0:
        #     x = F.pad(x, (0, 1, 0, 0), 'constant', 0)
        # x = padding(x, 2)
        # print('space-to-depth:', x.shape)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # 用值 val 填充输入张量
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)  ##
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # b, c, w, h = x.shape
        # if w % 2 == 0 and h % 2 != 0:
        #     x = F.pad(x, (0, 0, 0, 1), 'constant', 0)
        # elif w % 2 != 0 and h % 2 == 0:
        #     x = F.pad(x, (0, 1, 0, 0), 'constant', 0)
        # x = padding(x, 1)
        # print('Focus:', x.shape)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
