import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv

from mvseg3d.utils.spconv_utils import replace_feature


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs, batch_size, batch_indices):
        """Forward function."""
        ocr_context = []
        for i in range(batch_size):
            # [1, n, channels]
            feat = feats[batch_indices == i].unsqueeze(0)
            # [1, num_classes, n]
            prob = probs[batch_indices == i].unsqueeze(0)
            prob = prob.permute(0, 2, 1)
            prob = F.softmax(self.scale * prob, dim=2)
            # [1, num_classes, channels]
            context = torch.matmul(prob, feat).contiguous()
            ocr_context.append(context)
        # [batch_size, num_classes, channels]
        ocr_context = torch.cat(ocr_context)
        return ocr_context


class OCRLayer(nn.Module):
    def __init__(self, in_channels, nhead, scale=1., attn_drop=0.):
        super(OCRLayer, self).__init__()

        self.scale = scale
        self.transform_input = self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_gather_module = SpatialGatherModule(self.scale)
        self.attn = nn.MultiheadAttention(in_channels, nhead, dropout=attn_drop)
        self.bottleneck = nn.Linear(in_channels * 2, in_channels, bias=False)

    def forward(self, inputs, probs, batch_size):
        inputs = self.transform_input(inputs)
        feats = inputs.features
        batch_indices = inputs.indices[:, 0]
        ocr_context = self.spatial_gather_module(inputs.features, probs, batch_size, batch_indices)
        for i in range(batch_size):
            query_feat = inputs.features[batch_indices == i]
            value_feat = key_feat = ocr_context[i]
            attn_out, attn_weights = self.attn(query_feat, key_feat, value_feat)
            inputs.features[batch_indices == i] = attn_out
        feats = torch.cat([inputs.features, feats], dim=1)
        feats = self.bottleneck(feats)
        inputs = replace_feature(inputs, feats)
        return inputs
