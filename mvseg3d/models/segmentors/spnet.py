from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvseg3d.models.voxel_encoders import MeanVFE
from mvseg3d.models.backbones import SparseUnet
from mvseg3d.ops import voxel_to_point, voxel_max_pooling


class ImageAttentionLayer(nn.Module):
    def __init__(self, point_channels, image_channels):
        super(ImageAttentionLayer, self).__init__()
        hidden_dim = 32
        self.fc_image = nn.Sequential(nn.Linear(image_channels, image_channels, bias=False),
                                      nn.BatchNorm1d(image_channels),
                                      nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(point_channels, hidden_dim)
        self.fc2 = nn.Linear(image_channels, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, point_features, image_features):
        rp = self.fc1(point_features)
        ri = self.fc2(image_features)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp)))

        image_features = self.fc_image(image_features)
        out = image_features * att

        return out


class SPNet(nn.Module):
    def __init__(self, dataset):
        super(SPNet, self).__init__()

        dim_point = dataset.dim_point
        if dataset.use_cylinder:
            dim_point = dim_point + 2

        self.point_feature_channel = 64
        self.point_encoder = nn.Sequential(
            nn.BatchNorm1d(dim_point),
            nn.Linear(dim_point, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.point_feature_channel))

        self.use_image_feature = dataset.use_image_feature
        if self.use_image_feature:
            self.image_feature_channel = dataset.dim_image_feature
        else:
            self.image_feature_channel = 0

        self.use_multi_sweeps = dataset.use_multi_sweeps
        if self.use_multi_sweeps:
            self.mean_vfe = MeanVFE(dim_point)
            self.voxel_in_feature_channel = self.mean_vfe.voxel_feature_channel
        else:
            self.mean_vfe = None
            self.voxel_in_feature_channel = self.point_feature_channel + self.image_feature_channel

        self.voxel_feature_channel = 32
        self.voxel_encoder = SparseUnet(self.voxel_in_feature_channel,
                                        self.voxel_feature_channel,
                                        dataset.grid_size)

        if self.use_image_feature:
            self.ia_layer = ImageAttentionLayer(point_channels=self.point_feature_channel + self.voxel_feature_channel,
                                                image_channels=self.image_feature_channel)

        self.fusion_feature_channel = 64
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.point_feature_channel + self.voxel_feature_channel, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.fusion_feature_channel, bias=False),
            nn.BatchNorm1d(self.fusion_feature_channel),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(nn.Linear(self.fusion_feature_channel, 32, bias=False),
                                        nn.BatchNorm1d(32),
                                        nn.Dropout(0.1),
                                        nn.Linear(32, dataset.num_classes, bias=False))

        self.voxel_classifier = nn.Sequential(
            nn.Linear(self.voxel_feature_channel, 32, bias=False),
            nn.BatchNorm1d(32),
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
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, batch_dict):
        points = batch_dict['points'][:, 1:]
        point_per_features = self.point_encoder(points)

        # decorating points with pixel-level semantic score
        if self.use_image_feature:
            point_image_features = batch_dict['point_image_features']

        # encode voxel features
        point_voxel_ids = batch_dict['point_voxel_ids']
        if self.use_multi_sweeps:
            batch_dict['voxel_features'] = self.mean_vfe(batch_dict)
        else:
            if self.use_image_feature:
                batch_dict['voxel_features'] = voxel_max_pooling(torch.cat([point_per_features, point_image_features], dim=1), point_voxel_ids)
            else:
                batch_dict['voxel_features'] = voxel_max_pooling(point_per_features, point_voxel_ids)

        batch_dict = self.voxel_encoder(batch_dict)
        point_voxel_features = voxel_to_point(batch_dict['voxel_features'], point_voxel_ids)

        # attention fusion image features
        point_fusion_features = torch.cat([point_per_features, point_voxel_features], dim=1)
        if self.use_image_feature:
            point_image_features = self.ia_layer(point_per_features, point_image_features)
            point_fusion_features = torch.cat([point_fusion_features, point_image_features], dim=1)

        # feature encoder
        point_fusion_features = self.fusion_encoder(point_fusion_features)

        result = OrderedDict()
        point_out = self.classifier(point_fusion_features)
        result['point_out'] = point_out

        voxel_features = batch_dict['voxel_features']
        voxel_out = self.voxel_classifier(voxel_features)
        result['voxel_out'] = voxel_out

        return result
