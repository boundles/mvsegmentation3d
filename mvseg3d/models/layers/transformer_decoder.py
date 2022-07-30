from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEncodingSine


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        # (N, C) -> (N, 1, C)
        q, k, tgt = q.unsqueeze(1), k.unsqueeze(1), tgt.unsqueeze(1)
        attn_tgt = self.self_attn(q, k, value=tgt)[0]
        out = tgt + self.dropout(attn_tgt)
        out = self.norm(out)
        # (N, 1, C)->(N, C)
        out = out.squeeze(1)
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        # (N, C) -> (N, 1, C)
        q, k, memory = q.unsqueeze(1), k.unsqueeze(1), memory.unsqueeze(1)
        attn_tgt = self.self_attn(query=q, key=k, value=memory)[0]
        out = tgt + self.dropout(attn_tgt)
        out = self.norm(out)
        # (N, 1, C)->(N, C)
        out = out.squeeze(1)
        return out


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super(FFNLayer, self).__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        out = x + self.dropout(out)
        out = self.norm(out)
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
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
        outputs_mask = self.forward_prediction_heads(output, mask_features)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                pos=pos[level_index],
                query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_mask = self.forward_prediction_heads(output, mask_features)
            predictions_mask.append(outputs_mask)

        assert len(predictions_mask) == self.num_layers + 1

        return predictions_mask[-1]

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("qc,cn->qn", mask_embed, mask_features)

        return outputs_mask