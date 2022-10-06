from collections import OrderedDict

import torch
import torch.nn as nn

from mvseg3d.models.voxel_encoders import VFE
from mvseg3d.models.backbones import SparseUnet
from mvseg3d.models.layers import FlattenSELayer
from mvseg3d.ops import voxel_to_point


class SPNet(nn.Module):
    def __init__(self, dataset):
        super(SPNet, self).__init__()

        dim_point = dataset.dim_point
        if dataset.use_cylinder:
            dim_point = dim_point + 5
        else:
            dim_point = dim_point + 3

        self.point_feature_channel = 64
        self.point_encoder = nn.Sequential(
            nn.BatchNorm1d(dim_point),
            nn.Linear(dim_point, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.point_feature_channel))

        self.use_image_feature = dataset.use_image_feature
        if self.use_image_feature:
            self.point_feature_channel += dataset.dim_image_feature

        self.use_multi_sweeps = dataset.use_multi_sweeps
        if self.use_multi_sweeps:
            self.vfe = VFE(dim_point, reduce='mean')
        else:
            self.vfe = VFE(self.point_feature_channel, reduce='max')

        self.voxel_in_feature_channel = self.vfe.voxel_feature_channel
        self.voxel_feature_channel = 64
        self.voxel_encoder = SparseUnet(self.voxel_in_feature_channel,
                                        self.voxel_feature_channel,
                                        dataset.grid_size,
                                        dataset.voxel_size,
                                        dataset.point_cloud_range)

        self.fusion_feature_channel = 64
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.point_feature_channel + self.voxel_feature_channel, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.fusion_feature_channel, bias=False),
            nn.BatchNorm1d(self.fusion_feature_channel),
            nn.ReLU(inplace=True)
        )

        self.se = FlattenSELayer(self.fusion_feature_channel)

        self.classifier = nn.Sequential(nn.Linear(self.fusion_feature_channel, 64, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(True),
                                        nn.Dropout(0.1),
                                        nn.Linear(64, dataset.num_classes, bias=False))

        self.voxel_classifier = nn.Sequential(
            nn.Linear(self.voxel_feature_channel, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, dataset.num_classes, bias=False))

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, batch_dict):
        points = batch_dict['points'][:, 1:]
        if self.use_multi_sweeps:
            cur_point_indices = (points[:, 3] == 0)
            point_per_features = self.point_encoder(points[cur_point_indices])
        else:
            point_per_features = self.point_encoder(points)

        # decorating points with pixel-level semantic score
        if self.use_image_feature:
            point_image_features = batch_dict['point_image_features']
            point_per_features = torch.cat([point_per_features, point_image_features], dim=1)

        # encode voxel features
        point_voxel_ids = batch_dict['point_voxel_ids']
        if self.use_multi_sweeps:
            batch_dict['voxel_features'] = self.vfe(points, point_voxel_ids)
        else:
            batch_dict['voxel_features'] = self.vfe(point_per_features, point_voxel_ids)
        batch_dict = self.voxel_encoder(batch_dict)

        # point features from encoded voxel feature
        if self.use_multi_sweeps:
            point_voxel_features = voxel_to_point(batch_dict['voxel_features'], point_voxel_ids[cur_point_indices])
        else:
            point_voxel_features = voxel_to_point(batch_dict['voxel_features'], point_voxel_ids)

        # fusion voxel features
        point_fusion_features = torch.cat([point_per_features, point_voxel_features], dim=1)
        point_fusion_features = self.fusion_encoder(point_fusion_features)

        # se block for channel attention
        if self.use_multi_sweeps:
            point_batch_indices = batch_dict['points'][:, 0][cur_point_indices]
        else:
            point_batch_indices = batch_dict['points'][:, 0]
        point_fusion_features = point_fusion_features + self.se(point_fusion_features, point_batch_indices)

        result = OrderedDict()
        point_out = self.classifier(point_fusion_features)
        result['point_out'] = point_out

        voxel_features = batch_dict['voxel_features']
        voxel_out = self.voxel_classifier(voxel_features)
        result['voxel_out'] = voxel_out

        return result
