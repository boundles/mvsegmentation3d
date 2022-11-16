import torch.nn as nn


class PointTransformer(nn.Module):
    def __init__(self, in_channel):
        super(PointTransformer, self).__init__()
        self.in_channel = in_channel

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        return x0
