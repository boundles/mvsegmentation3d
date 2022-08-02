from functools import partial

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from mvseg3d.utils.spconv_utils import replace_feature, conv_norm_act
from mvseg3d.models.layers import FlattenSELayer, SALayer, MultiScaleTransformerDecoder


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, with_se=False,
                 with_sa=False, norm_fn=None, act_fn=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.act = act_fn
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

        # spatial and channel attention
        if with_se:
            self.se = FlattenSELayer(planes)
        else:
            self.se = None

        if with_sa:
            self.sa = SALayer(planes, indice_key=indice_key)
        else:
            self.sa = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.act(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.se is not None:
            out = replace_feature(out, self.se(out.features, out.indices[:, 0]))

        if self.sa is not None:
            out = self.sa(out)

        if self.downsample is not None:
            identity = replace_feature(x, self.downsample(x.features))

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act(out.features))

        return out


class UpBlock(spconv.SparseModule):
    def __init__(self, inplanes, planes, norm_fn, act_fn, conv_type, layer_id):
        super(UpBlock, self).__init__()
        block = conv_norm_act

        self.transform = SparseBasicBlock(inplanes, inplanes, norm_fn=norm_fn, act_fn=act_fn,
                                          indice_key='subm' + str(layer_id))

        self.bottleneck = block(2 * inplanes, inplanes, 3, padding=1, norm_fn=norm_fn, act_fn=act_fn,
                                indice_key='subm' + str(layer_id))

        if conv_type == 'inverseconv':
            self.out = block(inplanes, planes, 3, norm_fn=norm_fn, act_fn=act_fn,
                             conv_type=conv_type, indice_key='spconv' + str(layer_id))
        elif conv_type == 'subm':
            self.out = block(inplanes, planes, 3, padding=1, norm_fn=norm_fn, act_fn=act_fn,
                             conv_type=conv_type, indice_key='subm' + str(layer_id))
        else:
            raise NotImplementedError

    def forward(self, x_bottom, x_lateral):
        x = self.transform(x_bottom)
        x = replace_feature(x, torch.cat([x_lateral.features, x.features], dim=1))
        x = self.bottleneck(x)
        x = self.out(x)
        return x


class SparseUnet(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, output_channels, grid_size, voxel_size, point_cloud_range):
        super(SparseUnet, self).__init__()
        self.sparse_shape = grid_size[::-1]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        act_fn = nn.ReLU(inplace=True)
        block = conv_norm_act

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            act_fn
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm1')
        )

        # [1504, 1504, 72] -> [752, 752, 36]
        self.conv2 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv',
                  indice_key='spconv2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm2')
        )

        # [752, 752, 36] -> [376, 376, 18]
        self.conv3 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv',
                  indice_key='spconv3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, with_se=True, indice_key='subm3')
        )

        # [376, 376, 18] -> [188, 188, 9]
        self.conv4 = spconv.SparseSequential(
            block(128, 256, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv',
                  indice_key='spconv4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, act_fn=act_fn, with_se=True, indice_key='subm4')
        )

        # [188, 188, 9] -> [376, 376, 18]
        self.up4 = UpBlock(256, 128, norm_fn, act_fn, conv_type='inverseconv', layer_id=4)
        # [376, 376, 18] -> [752, 752, 36]
        self.up3 = UpBlock(128, 64, norm_fn, act_fn, conv_type='inverseconv', layer_id=3)
        # [752, 752, 36] -> [1504, 1504, 72]
        self.up2 = UpBlock(64, 32, norm_fn, act_fn, conv_type='inverseconv', layer_id=2)
        # [1504, 1504, 72] -> [1504, 1504, 72]
        self.up1 = UpBlock(32, 32, norm_fn, act_fn, conv_type='subm', layer_id=1)

        # transformer decoder
        self.transformer_decoder = MultiScaleTransformerDecoder(in_channels=[64, 32], hidden_dim=48,
                                                                num_queries=output_channels, nheads=4,
                                                                dim_feedforward=128, mask_dim=32, dec_layers=4)
        self.aux_voxel_feature_channel = 32

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, Cin)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                voxel_features: (num_voxels, Cout)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # encoder
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # decoder
        x_up4 = self.up4(x_conv4, x_conv4)
        x_up3 = self.up3(x_up4, x_conv3)
        x_up2 = self.up2(x_up3, x_conv2)
        x_up1 = self.up1(x_up2, x_conv1)

        # transformer decoder
        output_masks = []
        for i in range(batch_size):
            batch_features_2 = x_up2.features[x_up2.indices[:, 0] == i]
            batch_features_1 = x_up1.features[x_up1.indices[:, 0] == i]
            batch_indices_2 = x_up2.indices[x_up2.indices[:, 0] == i]
            batch_indices_1 = x_up1.indices[x_up1.indices[:, 0] == i]
            features = [batch_features_2, batch_features_1]
            indices = [batch_indices_2, batch_indices_1]
            mask_features = x_up1.features[x_up1.indices[:, 0] == i]
            mask_features = mask_features.transpose(0, 1)
            output_mask = self.transformer_decoder(features, indices, mask_features).transpose(0, 1)
            output_masks.append(output_mask)
        output_masks = torch.sigmoid(torch.cat(output_masks, dim=0))

        batch_dict['voxel_features'] = output_masks
        batch_dict['aux_voxel_features'] = x_up2.features
        return batch_dict
