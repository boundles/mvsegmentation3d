import torch
import torch.nn as nn

from mvseg3d.models.voxel_encoders import MeanVFE
from mvseg3d.models.backbones import SparseUnet
from mvseg3d.models.layers import FlattenELayer
from mvseg3d.ops import voxel_to_point

class SPNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.point_feature_channel = 32
        self.point_encoder = nn.Sequential(
            nn.Linear(dataset.dim_point, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.point_feature_channel, bias=False),
            nn.BatchNorm1d(self.point_feature_channel),
            nn.ReLU(inplace=True))

        self.use_image_feature = dataset.use_image_feature
        if self.use_image_feature:
            self.image_feature_channel = 32
            self.image_encoder = nn.Sequential(
                nn.Linear(dataset.dim_image_feature, self.image_feature_channel, bias=False),
                nn.BatchNorm1d(self.image_feature_channel),
                nn.ReLU(inplace=True))
        else:
            self.image_feature_channel = 0

        self.vfe = MeanVFE(dataset.dim_point)
        self.voxel_encoder = SparseUnet(dataset.dim_point,
                                        dataset.grid_size,
                                        dataset.voxel_size,
                                        dataset.point_cloud_range)

        self.fusion_in_channel = self.point_feature_channel + self.voxel_encoder.voxel_feature_channel + \
                                 self.image_feature_channel
        self.fusion_out_channel = 64
        self.fusion_encoder = nn.Sequential(nn.Linear(self.fusion_in_channel, self.fusion_out_channel, bias=False),
                                            nn.BatchNorm1d(self.fusion_out_channel),
                                            nn.ReLU(inplace=True))

        self.se = FlattenELayer(self.fusion_out_channel)

        self.cls_layers = nn.Sequential(nn.Linear(self.fusion_out_channel, self.fusion_out_channel, bias=False),
                                        nn.BatchNorm1d(self.fusion_out_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.fusion_out_channel, dataset.num_classes, bias=False))

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        points = batch_dict['points'][:, 1:]
        point_voxel_ids = batch_dict['point_voxel_ids']

        point_per_features = self.point_encoder(points)
        voxel_enc = self.voxel_encoder(self.vfe(batch_dict))
        point_voxel_features = voxel_to_point(voxel_enc['voxel_features'], point_voxel_ids)

        # fusion multi-view features
        if self.use_image_feature:
            point_image_features = batch_dict['point_image_features']
            point_image_features = self.image_encoder(point_image_features)
            point_fusion_features = torch.cat([point_voxel_features, point_per_features, point_image_features], dim=1)
        else:
            point_fusion_features = torch.cat([point_voxel_features, point_per_features], dim=1)
        point_fusion_features = self.fusion_encoder(point_fusion_features)

        # channel attention
        batch_indices = batch_dict['points'][:, 0]
        point_fusion_features = point_fusion_features + self.se(point_fusion_features, batch_indices)

        point_fusion_features = self.dropout(point_fusion_features)
        out = self.cls_layers(point_fusion_features)
        return out
