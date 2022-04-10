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
        out = self.vfe(batch_dict)
        out = self.net(out)

        point_features = voxel_to_point(out['voxel_features'], out['point_voxel_ids'])

        return point_features
