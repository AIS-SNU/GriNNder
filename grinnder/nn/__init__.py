"""GNN model definitions."""


def __getattr__(name):
    if name == "GriNNderModel":
        from grinnder.nn.base import GriNNderModel
        return GriNNderModel
    if name == "GCN":
        from grinnder.nn.gcn import GCN
        return GCN
    if name == "GAT":
        from grinnder.nn.gat import GAT
        return GAT
    raise AttributeError(f"module 'grinnder.nn' has no attribute {name!r}")


__all__ = ["GriNNderModel", "GCN", "GAT"]
