"""GAT model for GriNNder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, ModuleList

from torch_geometric.nn import GATConv

from grinnder.nn.base import GriNNderModel


class GAT(GriNNderModel):
    """Graph Attention Network.

    IMPORTANT: GAT uses bipartite input ``(x, x[:num_dst])`` where
    ``x`` is all source nodes and ``x[:num_dst]`` is destination nodes only.
    This is because attention scores are computed between source and
    destination nodes in the bipartite subgraph.

    Uses LayerNorm (not BatchNorm) for partition-invariant normalization.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Output dimension.
        num_layers: Number of GAT layers (>= 2).
        heads: Number of attention heads.
        dropout: Dropout rate.
        norm: Use layer normalization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int = 8,
        dropout: float = 0.6,
        norm: bool = False,
    ):
        super().__init__(num_layers)
        self.dropout = dropout
        self.heads = heads

        # add_self_loops=False because adjacency is bipartite (non-square)
        # Self-loops are already included in the subgraph construction.
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, add_self_loops=False)
            )
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, add_self_loops=False)
        )

        self.norms = None
        if norm:
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(LayerNorm(hidden_channels * heads))

    def forward_layer(self, layer: int, x: Tensor, adj) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Bipartite input: (all sources, destination nodes only)
        num_dst = adj.sparse_sizes()[0]
        x = self.convs[layer]((x, x[:num_dst]), adj)
        if layer < self.num_layers - 1:
            if self.norms is not None:
                x = self.norms[layer](x)
            x = F.elu(x)
        return x

    def layer_dims(self) -> List[int]:
        dims = [self.convs[0].in_channels]
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                dims.append(conv.out_channels * conv.heads)
            else:
                dims.append(conv.out_channels)
        return dims
