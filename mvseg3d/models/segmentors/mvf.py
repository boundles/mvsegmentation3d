import torch.nn as nn

from mvseg3d.models.voxel_encoders import MeanVFE
from mvseg3d.models.backbones import SparseUnet

class MVFNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.vfe = MeanVFE(6)
        self.net = SparseUnet(self.vfe.get_output_feature_dim(),
                              dataset.grid_size,
                              dataset.voxel_size,
                              dataset.point_cloud_range)

    def forward(self, batch_dict):
        x = self.vfe(batch_dict)
        x = self.net(x)
        return x
