import torch
import torch.nn as nn

from torch_scatter import scatter

from mvseg3d.utils.data_utils import farthest_point_sample, square_distance
from mvseg3d.utils.spconv_utils import replace_feature


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop_rate=0., proj_drop_rate=0.):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_embed_dim = embed_dim // num_heads
        self.scale = head_embed_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class ContextLayer(nn.Module):
    def __init__(self, planes, num_groups=1024, num_heads=4):
        super(ContextLayer, self).__init__()

        self.num_groups = num_groups
        self.transformer = Block(planes, num_heads)
        self.bottleneck = nn.Sequential(
            nn.Linear(2 * planes, planes, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward function.
        Args:
            x (SparseTensor): The input with features: shape (N, C)
        Returns:
            SparseTensor: The output with features: shape (N, C)
        """
        context_features = []
        indices = x.indices.long()
        features = x.features
        for i in range(x.batch_size):
            batch_indices = indices[indices[:, 0] == i][:, 1:].unsqueeze(0)
            group_idx = farthest_point_sample(batch_indices, self.num_groups).squeeze(0)
            point_to_group_dists = square_distance(batch_indices.float(), batch_indices[:, group_idx, :].float())
            point_to_group_idx = torch.argmin(point_to_group_dists, dim=2).squeeze()
            batch_features = features[indices[:, 0] == i]
            group_features = scatter(batch_features, point_to_group_idx, dim=0, reduce='mean').unsqueeze(0)
            group_features = self.transformer(group_features).squeeze()
            batch_context_features = group_features[point_to_group_idx, :]
            context_features.append(batch_context_features)
        context_features = torch.cat(context_features, dim=0)
        x = replace_feature(x, torch.cat([features, context_features], dim=1))
        x = replace_feature(x, self.bottleneck(x.features))
        return x
