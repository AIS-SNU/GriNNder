"""Tests for GNN model definitions."""

import torch
import pytest

pyg = pytest.importorskip("torch_geometric")
torch_sparse = pytest.importorskip("torch_sparse")
from torch_sparse import SparseTensor

from grinnder.nn import GCN, GAT


@pytest.fixture
def adj():
    """Small bipartite adjacency: 4 dst nodes, 7 total source nodes."""
    row = torch.tensor([0, 0, 1, 1, 2, 3, 3])
    col = torch.tensor([0, 4, 1, 5, 2, 3, 6])
    return SparseTensor(
        row=row, col=col, sparse_sizes=(4, 7), is_sorted=False,
    )


class TestGCN:
    def test_forward_layer(self, adj):
        model = GCN(16, 32, 7, num_layers=3)
        x = torch.randn(7, 16)
        out = model.forward_layer(0, x, adj)
        assert out.shape == (4, 32)

    def test_layer_dims(self):
        model = GCN(16, 32, 7, num_layers=3)
        dims = model.layer_dims()
        assert dims == [16, 32, 32, 7]
        assert len(dims) == model.num_layers + 1


class TestGAT:
    def test_forward_layer(self, adj):
        model = GAT(16, 4, 7, num_layers=2, heads=8)
        x = torch.randn(7, 16)
        out = model.forward_layer(0, x, adj)
        assert out.shape == (4, 32)  # 4 heads * 8 hidden

    def test_last_layer(self, adj):
        model = GAT(16, 4, 7, num_layers=2, heads=8)
        x = torch.randn(7, 32)
        out = model.forward_layer(1, x, adj)
        assert out.shape == (4, 7)  # concat=False for last layer
