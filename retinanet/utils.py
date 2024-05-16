import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from retinanet.SPD_Conv import space_to_depth


def padding(ts):
    b, c, h, w = ts.shape
    if w % 2 == 0 and h % 2 == 0:
        return ts
    elif w % 2 == 0 and h % 2 != 0:
        ts = F.pad(ts, (0, 0, 0, 1), 'constant', 0)
    elif w % 2 != 0 and h % 2 == 0:
        ts = F.pad(ts, (0, 1, 0, 0), 'constant', 0)
    return ts


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod  # staticmethod用于修饰类中的方法,使其可以在不创建类实例的情况下调用方法，这样做的好处是执行效率比较高。当然，也可以像一般的方法一样用实例调用该方法
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = padding(x)   ###
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#         #                        padding=1, bias=False)
#         if stride == 2:
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
#                                    padding=1, bias=False)
#             self.spd = space_to_depth()
#             self.bn2 = nn.BatchNorm2d(4 * planes)
#             self.relu2 = nn.ReLU(inplace=True)
#             self.conv3 = nn.Conv2d(4 * planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
#             self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
#         else:
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                    padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(planes)
#             self.relu2 = nn.ReLU(inplace=True)
#             self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
#             self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
#         # self.bn2 = nn.BatchNorm2d(planes)
#         # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         # self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inplanes != planes * Bottleneck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(planes * Bottleneck.expansion)
#             )
#
#     def forward(self, x):
#
#         residual = x
#         # residual = self.shortcut(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#
#         out = self.conv2(out)
#         if self.stride == 2:
#             out = self.spd(out)
#         out = self.bn2(out)
#         out = self.relu2(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
class Bottleneck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if stride == 2:
            self.layers2 = [
                nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
                space_to_depth(),  # the output of this will result in 4*out_channels
                nn.BatchNorm2d(4 * out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(4 * out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            ]

        else:
            self.layers2 = [
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            ]

        self.layers.extend(self.layers2)

        self.residual_function = torch.nn.Sequential(*self.layers)

        # self.residual_function = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
        # 	space_to_depth(),   # the output of this will result in 4*out_channels
        #     nn.BatchNorm2d(4*out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(4*out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        # )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes
