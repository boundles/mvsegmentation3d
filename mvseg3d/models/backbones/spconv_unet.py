from functools import partial

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from mvseg3d.utils.spconv_utils import replace_feature, conv_norm_act
from mvseg3d.models.layers import FlattenSELayer, SALayer


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
            self.sa = SALayer(indice_key=indice_key)
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

        if self.sa:
            out = self.sa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act(out.features))

        return out

class UpBlock(spconv.SparseModule):
    def __init__(self, inplanes, planes, norm_fn, act_fn, layer_id, conv_type='subm'):
        super().__init__()
        self.conv_t = SparseBasicBlock(inplanes, inplanes, norm_fn=norm_fn, act_fn=act_fn,
                                       indice_key='subm' + layer_id)
        self.conv_rc = nn.Sequential(nn.Linear(2 * inplanes, inplanes, bias=False),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU(inplace=True))
        if conv_type == 'inverseconv':
            self.conv_m = conv_norm_act(inplanes, inplanes, 3, norm_fn=norm_fn, act_fn=act_fn, padding=1,
                                        indice_key='subm' + layer_id)
            self.conv_out = conv_norm_act(inplanes, planes, 3, norm_fn=norm_fn, act_fn=act_fn, conv_type=conv_type,
                                          indice_key='spconv' + layer_id)
        elif conv_type == 'subm':
            self.conv_m = conv_norm_act(inplanes, inplanes, 3, norm_fn=norm_fn, act_fn=act_fn,
                                        indice_key='subm' + layer_id)
            self.conv_out = SparseBasicBlock(inplanes, planes, norm_fn=norm_fn, act_fn=act_fn, with_se=True,
                                             with_sa=True, indice_key='subm' + layer_id)
        else:
            raise NotImplementedError

    def forward(self, x_bottom, x_lateral):
        x = self.conv_t(x_bottom)
        x = replace_feature(x, self.conv_rc(torch.cat([x_lateral.features, x.features], dim=1)))
        x = self.conv_m(x)
        x = self.conv_out(x)
        return x

class SparseUnet(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, grid_size, voxel_size, point_cloud_range):
        super().__init__()
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
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

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(32, 64, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv', indice_key='spconv2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm2')
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(64, 128, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv', indice_key='spconv3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm3')
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(128, 256, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv', indice_key='spconv4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm4')
        )

        # decoder
        self.voxel_feature_channel = 32
        self.up4 = UpBlock(256, 128, norm_fn, act_fn, layer_id='4', conv_type='inverseconv')
        self.up3 = UpBlock(128, 64, norm_fn, act_fn, layer_id='3', conv_type='inverseconv')
        self.up2 = UpBlock(64, 32, norm_fn, act_fn, layer_id='2', conv_type='inverseconv')
        self.up1 = UpBlock(32, self.voxel_feature_channel, norm_fn, act_fn, layer_id='1', conv_type='subm')

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
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.up4(x_conv4, x_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.up3(x_up4, x_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.up2(x_up3, x_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.up1(x_up2, x_conv1)

        batch_dict['voxel_features'] = x_up1.features
        return batch_dict
