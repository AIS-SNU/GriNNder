"""Tests for Trainer: forward, backward, train_epoch end-to-end."""

import torch
import pytest

pyg = pytest.importorskip("torch_geometric")
torch_sparse = pytest.importorskip("torch_sparse")
grdpart = pytest.importorskip("grdpart")

from grinnder.config import GriNNderConfig
from grinnder.data.partition import build_partitioned_graph
from grinnder.engine.trainer import Trainer
from grinnder.engine.streams import StreamManager
from grinnder.nn.gcn import GCN
from grinnder.nn.gat import GAT


@pytest.fixture
def small_graph_data():
    """Small graph for quick trainer tests."""
    num_nodes = 50
    edges = set()
    for i in range(num_nodes):
        for j in range(max(0, i - 3), min(num_nodes, i + 4)):
            if i != j:
                edges.add((i, j))
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    x = torch.randn(num_nodes, 16)
    y = torch.randint(0, 4, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:35] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[35:42] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[42:] = True
    return edge_index, x, y, train_mask, val_mask, test_mask


@pytest.fixture
def partitioned_graph(small_graph_data, tmp_path):
    edge_index, x, y, train_mask, val_mask, test_mask = small_graph_data
    config = GriNNderConfig(
        num_parts=4,
        partitioner="grinnder",
        mode="grinnder",
        storage_dir=str(tmp_path / "storage"),
        cache_mode="auto",
    )
    return build_partitioned_graph(
        edge_index, x, y, train_mask, val_mask, test_mask, config
    )


def _grinnder_config(tmp_path, num_parts=4):
    return GriNNderConfig(
        num_parts=num_parts,
        partitioner="grinnder",
        mode="grinnder",
        storage_dir=str(tmp_path / "storage"),
        cache_mode="auto",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStreamManager:
    def test_creation(self):
        sm = StreamManager(pool_size=2, device="cuda:0")
        assert len(sm.h2d) == 2
        assert len(sm.d2h) == 2
        assert len(sm.act_h2d) == 2
        assert sm.compute is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTrainerInit:
    def test_creation(self, partitioned_graph, tmp_path):
        config = _grinnder_config(tmp_path)
        model = GCN(16, 32, 4, num_layers=2).cuda()
        trainer = Trainer(model, partitioned_graph, config)

        assert len(trainer.host_features) == 3  # layers 0, 1, 2
        assert len(trainer.host_gradients) == 2  # None, layer 1
        assert trainer.host_gradients[0] is None
        assert len(trainer.device_features) == 2  # layers 0, 1
        assert len(trainer.activations) == 2

    def test_prefill_features(self, partitioned_graph, tmp_path):
        config = _grinnder_config(tmp_path)
        model = GCN(16, 32, 4, num_layers=2).cuda()
        trainer = Trainer(model, partitioned_graph, config)

        # host_features[0] should be filled with initial features
        for pid in range(4):
            feat = trainer.host_features[0][pid]
            assert feat.abs().sum() > 0, f"Partition {pid} features not prefilled"

    def test_one_partition_uses_direct_fast_path(self, small_graph_data, tmp_path):
        edge_index, x, y, train_mask, val_mask, test_mask = small_graph_data
        config = _grinnder_config(tmp_path, num_parts=1)
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )
        model = GCN(16, 32, 4, num_layers=2).cuda()
        trainer = Trainer(model, graph, config)

        assert trainer._use_single_partition_fast_path()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTrainerTrainEpoch:
    def test_gcn_train_epoch(self, partitioned_graph, tmp_path):
        """GCN should complete one training epoch without errors."""
        config = _grinnder_config(tmp_path)
        model = GCN(16, 32, 4, num_layers=2, dropout=0.0).cuda()
        trainer = Trainer(model, partitioned_graph, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(optimizer, criterion)

        assert "loss" in metrics
        assert "val_acc" in metrics
        assert "test_acc" in metrics
        assert metrics["loss"] > 0

    def test_loss_decreases(self, partitioned_graph, tmp_path):
        """Loss should decrease over multiple epochs."""
        config = _grinnder_config(tmp_path)
        model = GCN(16, 32, 4, num_layers=2, dropout=0.0).cuda()
        trainer = Trainer(model, partitioned_graph, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        losses = []
        for _ in range(5):
            metrics = trainer.train_epoch(optimizer, criterion)
            losses.append(metrics["loss"])

        # Loss should generally decrease (check first vs last)
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"

    def test_gat_train_epoch(self, partitioned_graph, tmp_path):
        """GAT should complete training without errors."""
        config = _grinnder_config(tmp_path)
        model = GAT(16, 4, 4, num_layers=2, heads=4, dropout=0.0).cuda()
        trainer = Trainer(model, partitioned_graph, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(optimizer, criterion)
        assert metrics["loss"] > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTrainerEvaluate:
    def test_evaluate(self, partitioned_graph, tmp_path):
        config = _grinnder_config(tmp_path)
        model = GCN(16, 32, 4, num_layers=2).cuda()
        trainer = Trainer(model, partitioned_graph, config)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = trainer.evaluate(criterion)
        assert "val_acc" in metrics
        assert "test_acc" in metrics
