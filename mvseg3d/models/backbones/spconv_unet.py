from functools import partial

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from mvseg3d.utils.spconv_utils import replace_feature, conv_norm_act
from mvseg3d.models.layers import SELayer


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_fn=None, act_fn=None, with_se=False, indice_key=None):
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
        self.with_se = with_se
        if self.with_se:
            self.se = SELayer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.act(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.with_se:
            out = replace_feature(out, self.se(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act(out.features))

        return out

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
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, with_se=True, indice_key='subm3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, with_se=True, indice_key='subm3')
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(128, 128, 3, norm_fn=norm_fn, act_fn=act_fn, stride=2, padding=1, conv_type='spconv', indice_key='spconv4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, with_se=True, indice_key='subm4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, with_se=True, indice_key='subm4')
        )

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm4')
        self.conv_up_m4 = SparseBasicBlock(256, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm4')
        self.inv_conv4 = block(128, 128, 3, norm_fn=norm_fn, act_fn=act_fn, conv_type='inverseconv', indice_key='spconv4')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(128, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm3')
        self.conv_up_m3 = SparseBasicBlock(256, 128, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm3')
        self.inv_conv3 = block(128, 64, 3, norm_fn=norm_fn, act_fn=act_fn, conv_type='inverseconv', indice_key='spconv3')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(64, 64, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm2')
        self.conv_up_m2 = SparseBasicBlock(128, 64, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm2')
        self.inv_conv2 = block(64, 32, 3, norm_fn=norm_fn, act_fn=act_fn, conv_type='inverseconv', indice_key='spconv2')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(32, 32, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm1')
        self.conv_up_m1 = SparseBasicBlock(64, 32, norm_fn=norm_fn, act_fn=act_fn, indice_key='subm1')

        self.voxel_feature_channel = 32
        self.conv5 = spconv.SparseSequential(
            block(32, self.voxel_feature_channel, 3, norm_fn=norm_fn, act_fn=act_fn, padding=1, indice_key='subm1')
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2
        Returns:
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

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
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        batch_dict['voxel_features'] = x_up1.features
        return batch_dict
