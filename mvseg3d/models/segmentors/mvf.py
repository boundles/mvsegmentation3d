import torch.nn as nn

from mvseg3d.models.voxel_encoders import MeanVFE
from mvseg3d.models.backbones import SparseUnet
from mvseg3d.utils.voxel_point_utils import voxel_to_point

class MVFNet(nn.Module):
    def __init__(self, dataset, num_point_features):
        super().__init__()

        self.vfe = MeanVFE(num_point_features)
        self.net = SparseUnet(self.vfe.get_output_feature_dim(),
                              dataset.grid_size,
                              dataset.voxel_size,
                              dataset.point_cloud_range)

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.net(batch_dict)

        point_features = voxel_to_point(batch_dict['voxel_features'], batch_dict['point_voxel_ids'])

        return point_features
