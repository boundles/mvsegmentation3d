from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from mvseg3d.models.backbones import PointTransformer
from mvseg3d.models.layers import FlattenSELayer
from mvseg3d.ops import knn_query


class DeepFusionBlock(nn.Module):
    def __init__(self, lidar_channel, image_channel, hidden_channel, n_neighbors, attn_pdrop=0.5):
        super(DeepFusionBlock, self).__init__()

        self.lidar_channel = lidar_channel
        self.image_channel = image_channel
        self.n_neighbors = n_neighbors

        self.q_embedding = nn.Linear(lidar_channel, hidden_channel)
        self.k_embedding = nn.Linear(image_channel, hidden_channel)
        self.v_embedding = nn.Linear(image_channel, hidden_channel)

        self.attn_dropout = nn.Dropout(attn_pdrop)

        self.c_proj = nn.Linear(hidden_channel, image_channel)

    def forward(self, points, point_id_offset, lidar_features, image_features):
        q = self.q_embedding(lidar_features)
        k = self.k_embedding(image_features)
        v = self.v_embedding(image_features)

        knn_ids, _ = knn_query(self.n_neighbors, points, points, point_id_offset, point_id_offset)
        k = k[knn_ids.long()]
        attn = torch.einsum('nc,nkc->nk', q, k) / np.sqrt(q.shape[-1])

        invalid_mask = (torch.sum(image_features, dim=1) == 0)
        invalid_mask = invalid_mask[knn_ids.long()]
        attn[invalid_mask] = float('-inf')
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)

        v = v[knn_ids.long()]
        y = torch.einsum('nk,nkc->nc', attn, v)
        y = self.c_proj(y)
        return y


class Segformer(nn.Module):
    def __init__(self, dataset):
        super(Segformer, self).__init__()

        self.use_multi_sweeps = dataset.use_multi_sweeps

        dim_point = dataset.dim_point
        self.point_feature_channel = 32

        self.point_transformer = PointTransformer(dim_point)

        self.use_image_feature = dataset.use_image_feature
        if self.use_image_feature:
            self.image_feature_channel = dataset.dim_image_feature
            self.deep_fusion = DeepFusionBlock(self.point_feature_channel, self.image_feature_channel, 32, 16)
        else:
            self.image_feature_channel = 0

        self.fusion_feature_channel = 64
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.point_feature_channel + self.image_feature_channel, 512, bias=False),
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
        point_id_offset = batch_dict['point_id_offset'].int()
        if self.use_multi_sweeps:
            cur_point_indices = (points[:, 3] == 0)
            cur_points = points[cur_point_indices]
        else:
            cur_points = points

        point_per_features = self.point_transformer((cur_points[:, :3], cur_points[:, 3:], point_id_offset))

        # decorating points with pixel-level semantic score
        if self.use_image_feature:
            point_image_features = batch_dict['point_image_features']
            point_image_features = self.deep_fusion(cur_points.contiguous(), point_id_offset,
                                                    point_per_features, point_image_features)

        # fusion point features
        point_fusion_features = torch.cat([point_per_features, point_image_features], dim=1)
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

        return result
