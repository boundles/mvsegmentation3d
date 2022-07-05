import torch.nn as nn

import spconv.pytorch as spconv

from mvseg3d.utils.spconv_utils import replace_feature


class SALayer(nn.Module):
    def __init__(self, planes, indice_key):
        super(SALayer, self).__init__()
        self.conv = spconv.SubMConv3d(planes, 1, 7, padding=3, bias=False, indice_key=indice_key + 'sa')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function.
        Args:
            x (SparseTensor): The input with features: shape (N, C)
        Returns:
            SparseTensor: The output with features: shape (N, C)
        """
        out = self.conv(x)
        out = replace_feature(out, self.sigmoid(out.features))
        out = replace_feature(out, x.features * out.features)

        return out
