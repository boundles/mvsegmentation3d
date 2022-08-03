from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        # (N, C) -> (N, 1, C)
        q, k, tgt = q.unsqueeze(1), k.unsqueeze(1), tgt.unsqueeze(1)
        attn_tgt = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout(attn_tgt)
        tgt = self.norm(tgt)
        # (N, 1, C) -> (N, C)
        tgt = tgt.squeeze(1)
        return tgt

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        # (N, C) -> (N, 1, C)
        q, k, memory, tgt = q.unsqueeze(1), k.unsqueeze(1), memory.unsqueeze(1), tgt.unsqueeze(1)
        attn_tgt = self.self_attn(query=q, key=k, value=memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout(attn_tgt)
        tgt = self.norm(tgt)
        # (N, 1, C) -> (N, C)
        tgt = tgt.squeeze(1)
        return tgt

class KMeansCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_queries, dropout=0.0):
        super(KMeansCrossAttentionLayer, self).__init__()

        self.num_queries = num_queries

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            MLP(d_model, d_model, d_model, 3)
        )

        self.bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, cluster_centers, point_features):
        mask_embeddings = self.mlp(cluster_centers)
        pred_logits = torch.einsum("qc,nc->nq", mask_embeddings, point_features)
        clustering_result = torch.argmax(pred_logits, dim=1)
        clustering_result = torch.one_hot(clustering_result, num_classes=self.num_queries).to(point_features.device)
        cluster_memory = torch.einsum("nq,nc->qc", clustering_result, point_features)
        cluster_centers = cluster_centers + self.dropout(self.bottleneck(cluster_memory))
        return pred_logits, cluster_centers

class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super(FFNLayer, self).__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU(inplace=True)

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
