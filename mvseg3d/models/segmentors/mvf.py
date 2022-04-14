import torch
import torch.nn as nn

from mvseg3d.models.voxel_encoders import MeanVFE
from mvseg3d.models.backbones import SparseUnet
from mvseg3d.utils.voxel_point_utils import voxel_to_point

class MVFNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.point_feature_channel = 16
        self.point_encoder = nn.Sequential(nn.Linear(dataset.point_dim, self.point_feature_channel, bias=False),
                                           nn.BatchNorm1d(self.point_feature_channel),
                                           nn.ReLU(inplace=True))

        self.vfe = MeanVFE(dataset.point_dim)
        self.voxel_encoder = SparseUnet(self.vfe.voxel_feature_channel,
                                        dataset.grid_size,
                                        dataset.voxel_size,
                                        dataset.point_cloud_range)

        self.fusion_feature_channel = self.point_feature_channel + self.voxel_encoder.voxel_feature_channel
        self.cls_layers = nn.Sequential(nn.Linear(self.fusion_feature_channel, self.fusion_feature_channel, bias=False),
                                        nn.BatchNorm1d(self.fusion_feature_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.fusion_feature_channel, dataset.num_classes, bias=False))

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch_dict):
        point_per_features = self.point_encoder(batch_dict['points'])

        enc_out = self.voxel_encoder(self.vfe(batch_dict))
        point_voxel_features = voxel_to_point(enc_out['voxel_features'], enc_out['point_voxel_ids'])

        point_fusion_features = torch.cat([point_voxel_features, point_per_features], dim=1)
        out = self.cls_layers(point_fusion_features)

        if self.training:
            labels = batch_dict['labels']
            loss = self.ce_loss(out, labels)
            return out, loss
        else:
            return out
