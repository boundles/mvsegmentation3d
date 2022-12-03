import torch
import torch.nn as nn
import torch.nn.functional as F

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
            prob = F.softmax(self.scale * prob, dim=2)
            prob = prob.permute(0, 2, 1)
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
        self.spatial_gather_module = SpatialGatherModule(self.scale)
        self.attn = nn.MultiheadAttention(in_channels, nhead, dropout=attn_drop)
        self.bottleneck = nn.Linear(in_channels * 2, in_channels, bias=False)

    def forward(self, inputs, probs, batch_size):
        feats = inputs.features.clone()
        batch_indices = inputs.indices[:, 0]
        ocr_context = self.spatial_gather_module(feats, probs, batch_size, batch_indices)
        for i in range(batch_size):
            feat = feats[batch_indices == i]
            context = ocr_context[i]
            attn_out, attn_weights = self.attn(feat, context, context)
            feats[batch_indices == i] = attn_out
        feats = torch.cat([inputs.features, feats], dim=1)
        feats = self.bottleneck(feats)
        inputs = replace_feature(inputs, feats)
        return inputs
