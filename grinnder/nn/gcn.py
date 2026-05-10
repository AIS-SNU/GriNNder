"""GCN model for GriNNder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, ModuleList

from torch_geometric.nn import GCNConv

from grinnder.nn.base import GriNNderModel


class GCN(GriNNderModel):
    """Graph Convolutional Network.

    GCN normalization (D^{-1/2} A D^{-1/2}) is pre-computed on the full graph
    and stored as edge values in the partitioned adjacency. GCNConv uses these
    pre-computed values rather than recomputing per-subgraph.

    Uses LayerNorm (not BatchNorm) for partition-invariant normalization.
    LayerNorm normalizes per-node, so statistics are independent of which
    partition a node belongs to.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer dimension.
        out_channels: Output (num_classes) dimension.
        num_layers: Number of GCN layers (>= 2).
        dropout: Dropout rate.
        norm: Use layer normalization.
        residual: Use residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        norm: bool = False,
        residual: bool = False,
    ):
        super().__init__(num_layers)
        self.dropout = dropout
        self.residual = residual

        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.norms = None
        if norm:
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(LayerNorm(hidden_channels))

    def forward_layer(self, layer: int, x: Tensor, adj) -> Tensor:
        h = self.convs[layer](x, adj)
        if layer < self.num_layers - 1:
            if self.norms is not None:
                h = self.norms[layer](h)
            if self.residual and h.shape == x.shape:
                h = h + x
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def layer_dims(self) -> List[int]:
        return [self.convs[0].in_channels] + [c.out_channels for c in self.convs]
