import torch
import torch.nn as nn
from torch.nn import functional as F

from mvseg3d.utils.spconv_utils import replace_feature


class PPMLayer(nn.Module):
    def __init__(self, in_dim, sizes=(1, 2, 3, 6)):
        super(PPMLayer, self).__init__()
        self.features = []
        reduction_dim = int(in_dim / len(sizes))
        for size in sizes:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
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
        indices = x.indices
        x_dense = x.dense()
        b, c, h, w, l = x_dense.shape
        x_dense = x_dense.reshape(b, c * h, w, l)
        out = self.ppm(x_dense)
        out = out.reshape(b, c, h, w, l)
        out_features = out[indices[:, 0].long(), :, indices[:, 1].long(), indices[:, 2].long(), indices[:, 3].long()]
        out = replace_feature(out, out_features)

        return out
