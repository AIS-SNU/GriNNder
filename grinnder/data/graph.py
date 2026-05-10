"""PartitionedGraph: preprocessed graph ready for GriNNder training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class PartitionedGraph:
    """A graph that has been partitioned and preprocessed for GriNNder training.

    Created by ``build_partitioned_graph()``. Contains everything needed for
    training: per-partition adjacency (bipartite), boundaries, features, labels.

    Per-partition adjacency is bipartite:
        rows = destination (intra-partition) nodes [batch_size]
        cols = source (intra + boundary) nodes [batch_size + boundary_nodes]

    Adjacencies are stored on NVMe and loaded on-demand per partition.
    Boundaries are always in host RAM (small int64 index tensors).
    Features and labels are reordered by partition via ``perm``.
    """

    num_nodes: int
    num_edges: int
    num_parts: int
    partition_sizes: List[int]  # nodes per partition

    # Per-partition adjacency stored as (rowptr, col, value) tuples.
    # Stored on NVMe, loaded on-demand. value contains GCN norm coefficients.
    adj_csr: List[Tuple[Tensor, Tensor, Optional[Tensor]]]

    # Boundary indices: boundaries[pid][src_pid] = local indices in partition src_pid.
    # boundaries[pid][pid] = None. ALWAYS in host RAM.
    boundaries: List[List[Optional[Tensor]]]

    # Expanded sizes: total nodes per partition including boundary nodes.
    # expanded_sizes[pid] = partition_sizes[pid] + sum(boundary sizes)
    expanded_sizes: List[int]

    # Full graph data (reordered by partition)
    features: Any       # [num_nodes, feat_dim] tensor or lazy feature store
    labels: Tensor      # [num_nodes] on CPU
    train_mask: Tensor  # [num_nodes] bool
    val_mask: Tensor    # [num_nodes] bool
    test_mask: Tensor   # [num_nodes] bool

    # Partition metadata (from grdpart)
    perm: Tensor  # [num_nodes] long -- node reordering permutation
    ptr: Tensor   # [num_parts+1] long -- partition boundary pointers

    @property
    def feat_dim(self) -> int:
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.labels.max().item()) + 1

    def partition_features(self, pid: int) -> Tensor:
        """Get features for partition pid (intra-partition nodes only)."""
        start = int(self.ptr[pid].item())
        end = int(self.ptr[pid + 1].item())
        if hasattr(self.features, "partition"):
            return self.features.partition(start, end)
        return self.features[start:end]

    def partition_labels(self, pid: int) -> Tensor:
        """Get labels for partition pid."""
        return self.labels[self.ptr[pid] : self.ptr[pid + 1]]

    def partition_train_mask(self, pid: int) -> Tensor:
        """Get training mask for partition pid."""
        return self.train_mask[self.ptr[pid] : self.ptr[pid + 1]]

    def partition_val_mask(self, pid: int) -> Tensor:
        return self.val_mask[self.ptr[pid] : self.ptr[pid + 1]]

    def partition_test_mask(self, pid: int) -> Tensor:
        return self.test_mask[self.ptr[pid] : self.ptr[pid + 1]]
