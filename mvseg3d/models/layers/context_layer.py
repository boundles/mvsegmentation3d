import torch
import torch.nn as nn
from torch.nn import functional as F

from mvseg3d.utils.spconv_utils import replace_feature


class PPMLayer(nn.Module):
    def __init__(self, planes, sizes=(24, 48, 96)):
        super(PPMLayer, self).__init__()
        self.features = []
        reduction_dim = int(planes / len(sizes))
        for size in sizes:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(planes, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        ppm_outs = [x]
        for f in self.features:
            ppm_outs.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        ppm_outs = torch.cat(ppm_outs, 1)
        out = self.bottleneck(ppm_outs)
        return out


class ContextLayer(nn.Module):
    def __init__(self, planes):
        super(ContextLayer, self).__init__()
        self.ppm = PPMLayer(planes)

    def forward(self, x):
        """Forward function.
        Args:
            x (SparseTensor): The input with features: shape (N, C)
        Returns:
            SparseTensor: The output with features: shape (N, C)
        """
        indices = x.indices.long()
        x_dense = x.dense()
        b, c, h, w, l = x_dense.shape
        x_dense = x_dense.reshape(b, c * h, w, l)
        out_dense = self.ppm(x_dense)
        out_dense = out_dense.reshape(b, c, h, w, l)
        out_features = out_dense[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
        x = replace_feature(x, out_features)
        return x
