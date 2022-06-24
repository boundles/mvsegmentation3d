import torch
import torch.nn as nn

import spconv.pytorch as spconv
from mvseg3d.utils.spconv_utils import replace_feature


class SALayer(nn.Module):
    def __init__(self, indice_key):
        super(SALayer, self).__init__()
        self.conv = spconv.SubMConv3d(2, 1, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function.
        Args:
            x (SparseTensor): The input with features: shape (N, C) and indices: shape (N, 4)
        Returns:
            SparseTensor: The output with features: shape (N, C) and indices: shape (N, 4)
        """
        identity = x
        avg_out = torch.mean(x.features, dim=1, keepdim=True)
        max_out, _ = torch.max(x.features, dim=1, keepdim=True)
        out = replace_feature(x, torch.cat([avg_out, max_out], dim=1))
        out = self.conv(out)
        out = replace_feature(out, self.sigmoid(out.features))
        out = replace_feature(out, identity.features * out.features)

        return out
