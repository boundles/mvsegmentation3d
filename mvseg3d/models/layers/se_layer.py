import torch.nn as nn
from mvseg3d.ops import voxel_pooling


class FlattenELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(FlattenELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, indices):
        """Forward function.
        Args:
            x (torch.Tensor): The input shape (N, C) where C = dim of input
            indices (torch.Tensor): The indices shape (N,)
        Returns:
            torch.Tensor: The output with shape (N, C)
        """
        unique_ids, counts = indices.unique(return_counts=True)
        out = voxel_pooling(x, indices, counts)
        out = self.fc(out)
        out = out[indices]
        out = x * out
        return out
