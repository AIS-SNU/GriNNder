"""Tests for data pipeline (datasets, partitioning, graph)."""

import torch
import pytest

from grinnder.data.graph import PartitionedGraph

pyg = pytest.importorskip("torch_geometric")


class TestGCNNorm:
    def test_simple_graph(self):
        from grinnder.data.datasets import compute_gcn_norm

        # Triangle graph: 0-1, 1-2, 0-2
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]])
        weights = compute_gcn_norm(edge_index, num_nodes=3, add_self_loops=True)
        assert weights.shape[0] > 0
        assert (weights > 0).all()
        assert (weights <= 1).all()


class TestOGBCompatibility:
    def test_torch_load_context_sets_weights_only_false(self, monkeypatch):
        from grinnder.data.datasets import _torch_load_with_weights_only_disabled

        calls = []

        def fake_load(*args, **kwargs):
            calls.append(kwargs.copy())
            return "loaded"

        monkeypatch.setattr(torch, "load", fake_load)
        with _torch_load_with_weights_only_disabled():
            assert torch.load("processed.pt") == "loaded"

        assert calls == [{"weights_only": False}]
        assert torch.load is fake_load


class TestIGBHelpers:
    def test_igb_label_filenames(self):
        from grinnder.data.datasets import _igb_label_filename

        assert _igb_label_filename(19) == "node_label_19.npy"
        assert _igb_label_filename(2983) == "node_label_2K.npy"

        with pytest.raises(ValueError):
            _igb_label_filename(7)

    def test_igb_required_paths(self, tmp_path):
        from grinnder.data.datasets import _igb_required_paths

        paths = _igb_required_paths(tmp_path, "medium", 19)
        assert paths == [
            tmp_path / "medium" / "processed" / "paper" / "node_feat.npy",
            tmp_path / "medium" / "processed" / "paper" / "node_label_19.npy",
            tmp_path
            / "medium"
            / "processed"
            / "paper__cites__paper"
            / "edge_index.npy",
        ]


class TestPartitionedGraph:
    def test_partition_accessors(self):
        graph = PartitionedGraph(
            num_nodes=10,
            num_edges=20,
            num_parts=2,
            partition_sizes=[6, 4],
            adj_csr=[],
            boundaries=[],
            expanded_sizes=[8, 6],
            features=torch.randn(10, 3),
            labels=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            train_mask=torch.ones(10, dtype=torch.bool),
            val_mask=torch.zeros(10, dtype=torch.bool),
            test_mask=torch.zeros(10, dtype=torch.bool),
            perm=torch.arange(10),
            ptr=torch.tensor([0, 6, 10]),
        )

        assert graph.feat_dim == 3
        assert graph.num_classes == 2
        assert graph.partition_features(0).shape == (6, 3)
        assert graph.partition_features(1).shape == (4, 3)
        assert graph.partition_labels(0).shape == (6,)
        assert graph.partition_train_mask(0).sum() == 6
