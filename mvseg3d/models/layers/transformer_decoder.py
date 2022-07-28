import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # (N, C) -> (1, N, C)
        x = x.unsqueeze(0)
        q = k = v = x
        attn_output = self.self_attn(q, k, value=v)[0]
        out = x + self.dropout(attn_output)
        out = self.norm(out)
        # (1, N, C)->(N, C)
        out = out.squeeze(0)
        return out


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, q, kv):
        # (N, C) -> (1, N, C)
        q = q.unsqueeze(0)
        k = v = kv.unsqueeze(0)
        attn_output = self.self_attn(q, k, value=v)[0]
        out = q + self.dropout(attn_output)
        out = self.norm(out)
        # (1, N, C)->(N, C)
        out = out.squeeze(0)
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

    def forward(self, x, mask_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels

        src = []
        for i in range(self.num_feature_levels):
            src.append(self.input_proj[i](x[i]) + self.level_embed.weight[i][None, :])

        predictions_mask = []

        # prediction heads on learnable query features
        output = self.query_feat.weight
        outputs_mask = self.forward_prediction_heads(output, mask_features)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](output, src[level_index])

            output = self.transformer_self_attention_layers[i](output)

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