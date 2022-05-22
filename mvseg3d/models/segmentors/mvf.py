import torch
import torch.nn as nn

from mvseg3d.models.voxel_encoders import MeanVFE
from mvseg3d.models.backbones import SparseUnet
from mvseg3d.utils.voxel_point_utils import voxel_to_point
from mvseg3d.models.losses.focal_loss import FocalLoss

class MVFNet(nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.point_feature_channel = 32
        self.point_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dataset.dim_point, 32, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Linear(32, 64, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Linear(64, self.point_feature_channel, bias=False),
                nn.BatchNorm1d(self.point_feature_channel),
                nn.ReLU(inplace=True)
            )
        ])

        self.image_feature_channel = 32
        self.image_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dataset.dim_image_feature, self.image_feature_channel, bias=False),
                nn.BatchNorm1d(self.image_feature_channel),
                nn.ReLU(inplace=True)
            )
        ])

        self.vfe = MeanVFE(dataset.dim_point)
        self.voxel_encoder = SparseUnet(self.vfe.voxel_feature_channel,
                                        dataset.grid_size,
                                        dataset.voxel_size,
                                        dataset.point_cloud_range)

        self.fusion_feature_channel = self.point_feature_channel + self.voxel_encoder.voxel_feature_channel + \
                                      self.image_feature_channel
        self.cls_layers = nn.Sequential(nn.Linear(self.fusion_feature_channel, self.fusion_feature_channel, bias=False),
                                        nn.BatchNorm1d(self.fusion_feature_channel),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.fusion_feature_channel, dataset.num_classes, bias=False))

        self.ce_loss = nn.CrossEntropyLoss(weight=dataset.class_weight, ignore_index=255)

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        point_per_features = self.point_encoder(batch_dict['points'])
        enc_out = self.voxel_encoder(self.vfe(batch_dict))
        point_voxel_features = voxel_to_point(enc_out['voxel_features'], enc_out['point_voxel_ids'])

        point_image_features = batch_dict['point_image_features']
        point_image_features = self.image_encoder(point_image_features)

        point_fusion_features = torch.cat([point_voxel_features, point_per_features, point_image_features], dim=1)
        point_fusion_features = self.dropout(point_fusion_features)

        out = self.cls_layers(point_fusion_features)

        if 'labels' in batch_dict:
            labels = batch_dict['labels']
            loss = self.ce_loss(out, labels)
            return out, loss
        else:
            return out
