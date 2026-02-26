"""
Parts of model definition adapted from this https://github.com/krishnanlab/cone 
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn
from torch import Tensor
from torch_geometric.nn import GINConv, GCNConv


class GNNLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        *,
        dropout: float = 0.0,
        layer_type: str = "gcn",
    ):
        super().__init__()

        self.layer_type = layer_type

        # GIN or GCN
        if layer_type == "gin":
            # GIN requires an MLP
            mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
            self.conv = GINConv(mlp, train_eps=True)
        elif layer_type == "gcn":
            self.conv = GCNConv(dim, dim)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        self.norm = pygnn.LayerNorm(dim, mode="node")
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.norm(x)

        # GCN supports edge_weight, GIN does not
        if self.layer_type == "gcn" and edge_weight is not None:
            x = self.conv(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv(x, edge_index)

        x = self.act(x)
        x = self.dropout(x)
        return x


class GNN(nn.Module):

    def __init__(
        self,
        dim: int,
        num_layers: int,
        *,
        num_embs: Optional[int] = None,
        dropout: float = 0.0,
        layer_type: str = "gcn",
    ):
        super().__init__()
        self.num_layers = num_layers

        # Raw embeddings
        self.emb = nn.Embedding(num_embs, dim)

        # Message-passing layers
        self.mps = nn.ModuleList()
        for i in range(num_layers):
            self.mps.append(GNNLayer(dim, dropout=dropout, layer_type=layer_type))

        # Post-message-passing processor
        self.post_mp = torch.nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        modules = self.modules()
        next(modules)  # skip self
        for m in modules:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        cond_emb: Optional[Tensor] = None,
        return_intermediate: bool = False,
    ) -> Tuple[Tensor, List[Tensor]] | Tensor:
        x = self.emb.weight
        if cond_emb is not None:
            x = x + cond_emb

        if return_intermediate:
            intermediate = [x]

        for i, mp in enumerate(self.mps):
            # Message passing with residual connection
            x = mp(x, edge_index, edge_weight=edge_weight) + x
            if return_intermediate:
                intermediate.append(x)

        out = self.post_mp(x)

        if return_intermediate:
            return out, intermediate
        return out

















