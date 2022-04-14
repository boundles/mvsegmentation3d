import torch
import torch.nn as nn


class MeanVFE(nn.Module):
    def __init__(self, voxel_feature_channel):
        super().__init__()
        self._voxel_feature_channel = voxel_feature_channel

    @property
    def voxel_feature_channel(self):
        return self._voxel_feature_channel

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_num_points, C)
                voxel_num_points: optional (num_voxels)
        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict
