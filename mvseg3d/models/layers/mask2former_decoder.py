import torch
from torch import nn

from .position_encoding import PositionEncodingSine

from mvseg3d.utils.transformer_utils import MLP, FFNLayer, SelfAttentionLayer, CrossAttentionLayer


class MultiScaleTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            dec_layers: number of Transformer decoder layers
            mask_dim: mask feature dimension
        """
        super(MultiScaleTransformerDecoder, self).__init__()

        self.pe_layer = PositionEncodingSine(hidden_dim)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 2 scales)
        self.num_feature_levels = 2
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            if in_channels[i] != hidden_dim:
                self.input_proj.append(nn.Linear(in_channels[i], hidden_dim, bias=False))
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, features, indices, mask_features):
        # x is a list of multi-scale feature
        assert len(features) == self.num_feature_levels

        src = []
        pos = []
        for i in range(self.num_feature_levels):
            pos.append(self.pe_layer(indices[i]))
            src.append(self.input_proj[i](features[i]) + self.level_embed.weight[i][None, :])

        query_embed = self.query_embed.weight
        output = self.query_feat.weight

        predictions_mask = []

        # prediction heads on learnable query features
        outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                pos=pos[level_index],
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
            predictions_mask.append(outputs_mask)

        assert len(predictions_mask) == self.num_layers + 1

        return predictions_mask[-1]

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("qc,cn->qn", mask_embed, mask_features)

        # [Q, N] -> [B, Q, N] -> [B*h, Q, N]
        attn_mask = (outputs_mask.sigmoid().unsqueeze(0).repeat(self.num_heads, 1, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_mask, attn_mask
