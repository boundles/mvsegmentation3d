from typing import List

import torch
from torch import nn

from mvseg3d.utils.transformer_utils import MLP, FFNLayer, SelfAttentionLayer, KMeansCrossAttentionLayer


class KMaXTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            num_blocks: List[int],
            mask_dim: int
    ):
        """KMaXTransformerDecoder
        Args:
            in_channels: channels of the input features
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            num_blocks: A list of three integers specifying number of blocks for each stage
            mask_dim: mask feature dimension
        """
        super(KMaXTransformerDecoder, self).__init__()

        # define Transformer decoder here
        self.num_queries = num_queries
        self.num_heads = nheads
        self.num_blocks = num_blocks
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                KMeansCrossAttentionLayer(
                    d_model=hidden_dim,
                    num_queries=self.num_queries,
                    dropout=0.0
                )
            )

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
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

        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels):
            self.input_proj.append(nn.Linear(in_channels[i], hidden_dim, bias=False))

        # prediction heads
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, decoder_features, output_features):
        # x is a list of multi-scale feature
        assert len(decoder_features) == self.num_feature_levels

        src = []
        for i in range(self.num_feature_levels):
            src.append(self.input_proj[i](decoder_features[i]))

        # prediction heads on learnable query features
        aux_predictions_logits_list = []
        cluster_centers = self.query_feat.weight

        for i in range(self.num_feature_levels):
            num_block = self.num_blocks[i]
            for _ in range(num_block):
                # attention: cross-attention first
                predictions_logits, cluster_centers = self.transformer_cross_attention_layers[i](cluster_centers, src[i])

                cluster_centers = self.transformer_self_attention_layers[i](cluster_centers)

                # FFN
                cluster_centers = self.transformer_ffn_layers[i](cluster_centers)

                aux_predictions_logits_list.append(predictions_logits)

        assert len(aux_predictions_logits_list) == self.num_layers

        predictions_logits = self.forward_prediction_heads(cluster_centers, output_features)

        return predictions_logits, aux_predictions_logits_list

    def forward_prediction_heads(self, cluster_centers, point_features):
        mask_embeddings = self.decoder_norm(cluster_centers)
        mask_embeddings = self.mask_embed(mask_embeddings)
        predictions_logits = torch.einsum("qc,nc->nq", mask_embeddings, point_features)

        return predictions_logits