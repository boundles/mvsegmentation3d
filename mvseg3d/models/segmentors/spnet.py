from collections import OrderedDict

import torch
import torch.nn as nn

from mvseg3d.models.backbones import SparseUnet
from mvseg3d.models.layers import FlattenSELayer
from mvseg3d.ops import voxel_to_point, voxel_max_pooling


class SPNet(nn.Module):
    def __init__(self, dataset):
        super(SPNet, self).__init__()

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
            self.point_feature_channel += dataset.dim_image_feature

        self.voxel_feature_channel = 32
        self.voxel_encoder = SparseUnet(self.point_feature_channel,
                                        self.voxel_feature_channel,
                                        dataset.grid_size,
                                        dataset.voxel_size,
                                        dataset.point_cloud_range)

        self.fusion_feature_channel = 64
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.point_feature_channel + self.voxel_feature_channel, self.fusion_feature_channel, bias=False),
            nn.BatchNorm1d(self.fusion_feature_channel),
            nn.ReLU(inplace=True)
        )

        self.se = FlattenSELayer(self.fusion_feature_channel)

        self.aux_classifiers = nn.ModuleList()
        for i in range(1):
            self.aux_classifiers.append(nn.Sequential(
                nn.Linear(self.voxel_feature_channel, 32, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(32, dataset.num_classes, bias=False))
            )

        self.classifier = nn.Sequential(nn.Linear(self.fusion_feature_channel, 32, bias=False),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.1),
                                        nn.Linear(32, dataset.num_classes, bias=False))

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

    def forward(self, batch_dict):
        points = batch_dict['points'][:, 1:]
        point_voxel_ids = batch_dict['point_voxel_ids']

        point_per_features = self.point_encoder(points)
        if self.use_image_feature:
            point_image_features = batch_dict['point_image_features']
            point_per_features = torch.cat([point_per_features, point_image_features], dim=1)

        batch_dict = voxel_max_pooling(point_per_features, batch_dict)
        batch_dict = self.voxel_encoder(batch_dict)
        point_voxel_features = voxel_to_point(batch_dict['voxel_features'], point_voxel_ids)

        # fusion multi-view features
        point_fusion_features = torch.cat([point_voxel_features, point_per_features], dim=1)
        point_fusion_features = self.fusion_encoder(point_fusion_features)

        # channel attention
        batch_indices = batch_dict['points'][:, 0]
        point_fusion_features = point_fusion_features + self.se(point_fusion_features, batch_indices)

        result = OrderedDict()
        out = self.classifier(point_fusion_features)
        result['out'] = out

        aux_outs = []
        aux_voxel_features = batch_dict['aux_voxel_features']
        for idx, aux_voxel_feature in enumerate(aux_voxel_features):
            aux_out = self.aux_classifiers[idx](aux_voxel_feature)
            aux_outs.append(aux_out)
        result['aux_outs'] = aux_outs

        return result
