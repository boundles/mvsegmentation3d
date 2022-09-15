import torch.nn as nn

import spconv

from mvseg3d.utils.spconv_utils import replace_feature


class ContextLayer(nn.Module):
    def __init__(self, in_channels, channels, stride=1, indice_key=None):
        super(ContextLayer, self).__init__()
        self.conv1 = spconv.SubMConv3d(in_channels, channels, kernel_size=(3, 1, 1), stride=stride,
                                       padding=(1, 0, 0), bias=False, indice_key=indice_key)
        self.bn0 = nn.BatchNorm1d(channels)
        self.act1 = nn.Sigmoid()

        self.conv2 = spconv.SubMConv3d(in_channels, channels, kernel_size=(1, 3, 1), stride=stride,
                                       padding=(0, 1, 0), bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.Sigmoid()

        self.conv3 = spconv.SubMConv3d(in_channels, channels, kernel_size=(1, 1, 3), stride=stride,
                                       padding=(0, 0, 1), bias=False, indice_key=indice_key)
        self.bn3 = nn.BatchNorm1d(channels)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = replace_feature(out1, self.bn0(out1.features))
        out1 = replace_feature(out1, self.act1(out1.features))

        out2 = self.conv2(x)
        out2 = replace_feature(out2, self.bn2(out2.features))
        out2 = replace_feature(out2, self.act2(out2.features))

        out3 = self.conv3(x)
        out3 = replace_feature(out3, self.bn3(out3.features))
        out3 = replace_feature(out3, self.act3(out3.features))

        out = replace_feature(out1, out1.features + out2.features + out3.features)
        out = replace_feature(out, out.features * x.features)

        return out
