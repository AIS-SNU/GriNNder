"""GriNNderModel: base class for GNN models with GriNNder SSO."""

from __future__ import annotations

from abc import abstractmethod
from typing import List

import torch
from torch import Tensor


class GriNNderModel(torch.nn.Module):
    """Base class for GNN models that use GriNNder's SSO framework.

    Users subclass this and implement ``forward_layer()``. The Trainer
    handles all storage offloading, caching, and gradient management.

    The model only defines the per-layer computation. All buffer management,
    stream management, and I/O orchestration is handled by the Trainer.

    Args:
        num_layers: Number of GNN layers.

    Example::

        class MyGCN(GriNNderModel):
            def __init__(self, in_ch, hid_ch, out_ch, num_layers):
                super().__init__(num_layers)
                self.convs = ModuleList([GCNConv(in_ch, hid_ch)])
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hid_ch, hid_ch))
                self.convs.append(GCNConv(hid_ch, out_ch))

            def forward_layer(self, layer, x, adj):
                x = self.convs[layer](x, adj)
                if layer < self.num_layers - 1:
                    x = x.relu()
                return x

            def layer_dims(self):
                return [self.convs[0].in_channels] + \\
                       [c.out_channels for c in self.convs]
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self._num_layers = num_layers

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @abstractmethod
    def forward_layer(self, layer: int, x: Tensor, adj) -> Tensor:
        """Process one GNN layer for one partition.

        Args:
            layer: Layer index (0 to num_layers - 1).
            x: Gathered source features [batch_size + boundary_nodes, dim].
            adj: SparseTensor adjacency (bipartite).
                 Shape: [batch_size, batch_size + boundary_nodes].

        Returns:
            Output features [batch_size, out_dim] (destination nodes only).
        """
        ...

    @abstractmethod
    def layer_dims(self) -> List[int]:
        """Return feature dimensions: [in_dim, hid_dim, ..., out_dim].

        Length must be num_layers + 1. Used by Trainer to allocate buffers.
        """
        ...
