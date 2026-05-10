"""Data loading and graph partitioning."""


def __getattr__(name):
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
    if name == "NumpyFeatureStore":
        from grinnder.data.datasets import NumpyFeatureStore
        return NumpyFeatureStore
    if name == "compute_gcn_norm":
        from grinnder.data.datasets import compute_gcn_norm
        return compute_gcn_norm
    raise AttributeError(f"module 'grinnder.data' has no attribute {name!r}")


__all__ = [
    "PartitionedGraph",
    "build_partitioned_graph",
    "load_dataset",
    "load_igb",
    "NumpyFeatureStore",
    "compute_gcn_norm",
]
