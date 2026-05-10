"""GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading."""

__version__ = "0.1.0"

# Lazy imports to avoid requiring torch_geometric at import time.
# This allows `from grinnder.config import GriNNderConfig` to work
# without torch_geometric installed.


def __getattr__(name):
    if name == "GriNNderConfig":
        from grinnder.config import GriNNderConfig
        return GriNNderConfig
    if name == "GriNNderModel":
        from grinnder.nn.base import GriNNderModel
        return GriNNderModel
    if name == "GCN":
        from grinnder.nn.gcn import GCN
        return GCN
    if name == "GAT":
        from grinnder.nn.gat import GAT
        return GAT
    if name == "Trainer":
        from grinnder.engine.trainer import Trainer
        return Trainer
    if name == "PartitionedGraph":
        from grinnder.data.graph import PartitionedGraph
        return PartitionedGraph
    if name == "build_partitioned_graph":
        from grinnder.data.partition import build_partitioned_graph
        return build_partitioned_graph
    if name == "load_dataset":
        from grinnder.data.datasets import load_dataset
        return load_dataset
    if name == "load_igb":
        from grinnder.data.datasets import load_igb
        return load_igb
    raise AttributeError(f"module 'grinnder' has no attribute {name!r}")


__all__ = [
    "GriNNderConfig",
    "GriNNderModel",
    "GCN",
    "GAT",
    "Trainer",
    "PartitionedGraph",
    "build_partitioned_graph",
    "load_dataset",
    "load_igb",
]
