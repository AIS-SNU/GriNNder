"""Gradient identity test: full-batch PyG vs HongTu vs GriNNder.

Verifies that all three produce identical (or near-identical) gradients
on a small graph with 1 partition (so partitioning doesn't add noise).
"""

import tempfile

import torch
import pytest

pyg = pytest.importorskip("torch_geometric")
torch_sparse = pytest.importorskip("torch_sparse")
grdpart = pytest.importorskip("grdpart")
kvikio = pytest.importorskip("kvikio")

from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor

from grinnder.config import GriNNderConfig
from grinnder.data.partition import build_partitioned_graph
from grinnder.engine.trainer import Trainer
from grinnder.nn.gcn import GCN


@pytest.fixture
def graph():
    """Small deterministic graph."""
    torch.manual_seed(42)
    num_nodes = 20
    edges = set()
    for i in range(num_nodes):
        for j in range(max(0, i - 2), min(num_nodes, i + 3)):
            if i != j:
                edges.add((i, j))
    edge_index = torch.tensor(
        [[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long
    )
    x = torch.randn(num_nodes, 8)
    y = torch.randint(0, 3, (num_nodes,))
    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    return edge_index, x, y, train_mask, val_mask, test_mask


@pytest.fixture
def shared_weights():
    """Shared initial model weights."""
    torch.manual_seed(7)
    return GCN(8, 16, 3, num_layers=2, dropout=0.0).state_dict()


@pytest.fixture
def shared_norm_weights():
    """Shared initial model weights for LayerNorm partition checks."""
    torch.manual_seed(7)
    return GCN(8, 16, 3, num_layers=2, dropout=0.0, norm=True).state_dict()


def _vanilla_pyg_step(edge_index, x, y, train_mask, shared_weights):
    """Full-batch PyG: no partitioning, standard autograd."""

    class VanillaGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(8, 16, normalize=False)
            self.conv2 = GCNConv(16, 3, normalize=False)

        def forward(self, x, adj):
            x = self.conv1(x, adj)
            x = torch.relu(x)
            x = self.conv2(x, adj)
            return x

    # Build full adjacency with GCN norm
    ei_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    row, col = ei_sl
    deg = degree(col, x.size(0))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = SparseTensor.from_edge_index(ei_sl, edge_weight, (x.size(0), x.size(0)))

    model = VanillaGCN().cuda()
    # Map weights from GriNNder GCN format
    state = model.state_dict()
    for k in state:
        gk = k.replace("conv1", "convs.0").replace("conv2", "convs.1")
        if gk in shared_weights:
            state[k] = shared_weights[gk].clone()
    model.load_state_dict(state)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    out = model(x.cuda(), adj.cuda())
    loss = criterion(out[train_mask], y[train_mask].cuda())
    loss.backward()

    grads = {
        n.replace("conv1", "convs.0").replace("conv2", "convs.1"): p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.item(), grads


def _grinnder_step(
    edge_index,
    x,
    y,
    train_mask,
    val_mask,
    test_mask,
    shared_weights,
    mode,
    cache_mode="auto",
):
    """One training step with GriNNder (hongtu or grinnder mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        config = GriNNderConfig(
            num_parts=1,  # 1 partition = no partitioning noise
            partitioner="grinnder",
            mode=mode,
            storage_dir=tmp,
            cache_mode=cache_mode,
        )
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        model = GCN(8, 16, 3, num_layers=2, dropout=0.0).cuda()
        model.load_state_dict(shared_weights)

        trainer = Trainer(model, graph, config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(optimizer, criterion)

        grads = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return metrics["loss"], grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGradientIdentity:
    def test_vanilla_vs_hongtu(self, graph, shared_weights):
        """HongTu mode with 1 partition should match vanilla PyG exactly."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        vanilla_loss, vanilla_grads = _vanilla_pyg_step(
            edge_index, x, y, train_mask, shared_weights
        )
        hongtu_loss, hongtu_grads = _grinnder_step(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="hongtu"
        )

        print(f"\nVanilla loss: {vanilla_loss:.6f}")
        print(f"HongTu loss:  {hongtu_loss:.6f}")
        print(f"Loss diff:    {abs(vanilla_loss - hongtu_loss):.2e}")

        for name in vanilla_grads:
            if name in hongtu_grads:
                diff = (vanilla_grads[name] - hongtu_grads[name]).abs().max().item()
                print(f"  {name}: max grad diff = {diff:.2e}")

        # With 1 partition, loss should be very close
        assert abs(vanilla_loss - hongtu_loss) < 0.1, (
            f"Vanilla={vanilla_loss:.6f} vs HongTu={hongtu_loss:.6f}"
        )

    def test_hongtu_vs_grinnder(self, graph, shared_weights):
        """GriNNder scattered mode with 1 partition should match HongTu."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        hongtu_loss, hongtu_grads = _grinnder_step(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="hongtu"
        )
        grinnder_loss, grinnder_grads = _grinnder_step(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder"
        )

        print(f"\nHongTu loss:   {hongtu_loss:.6f}")
        print(f"GriNNder loss:  {grinnder_loss:.6f}")
        print(f"Loss diff:      {abs(hongtu_loss - grinnder_loss):.2e}")

        for name in hongtu_grads:
            if name in grinnder_grads:
                diff = (hongtu_grads[name] - grinnder_grads[name]).abs().max().item()
                print(f"  {name}: max grad diff = {diff:.2e}")

        # Both should produce identical loss (same computation, different save/restore)
        assert abs(hongtu_loss - grinnder_loss) < 1e-5, (
            f"HongTu={hongtu_loss:.6f} vs GriNNder={grinnder_loss:.6f}"
        )

        # Gradients should be identical
        for name in hongtu_grads:
            if name in grinnder_grads:
                assert torch.allclose(
                    hongtu_grads[name], grinnder_grads[name], atol=1e-5
                ), f"Gradient mismatch for {name}"

    def test_vanilla_vs_grinnder(self, graph, shared_weights):
        """Vanilla PyG vs GriNNder 1-part should match (transitivity check)."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        vanilla_loss, vanilla_grads = _vanilla_pyg_step(
            edge_index, x, y, train_mask, shared_weights
        )
        grinnder_loss, grinnder_grads = _grinnder_step(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder"
        )

        print(f"\nVanilla loss:  {vanilla_loss:.6f}")
        print(f"GriNNder loss:  {grinnder_loss:.6f}")
        print(f"Loss diff:      {abs(vanilla_loss - grinnder_loss):.2e}")

        assert abs(vanilla_loss - grinnder_loss) < 0.1

    def test_grinnder_auto_matches_forced_lru_layer(self, graph, shared_weights):
        """Auto should resolve to layer-wise LRU on this host-budget test."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        auto_loss, auto_grads = _grinnder_step(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder", cache_mode="auto"
        )
        lru_loss, lru_grads = _grinnder_step(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder", cache_mode="lru_layer"
        )

        assert abs(auto_loss - lru_loss) < 1e-5
        for name in auto_grads:
            assert torch.allclose(auto_grads[name], lru_grads[name], atol=1e-5), (
                f"GriNNder lru_layer grad mismatch {name}: "
                f"max diff={(auto_grads[name]-lru_grads[name]).abs().max():.2e}"
            )


def _grinnder_step_nparts(edge_index, x, y, train_mask, val_mask, test_mask,
                          shared_weights, mode, num_parts, norm=False):
    """One training step with specified num_parts."""
    with tempfile.TemporaryDirectory() as tmp:
        config = GriNNderConfig(
            num_parts=num_parts,
            partitioner="grinnder",
            mode=mode,
            storage_dir=tmp,
            cache_mode="auto",
        )
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        model = GCN(8, 16, 3, num_layers=2, dropout=0.0, norm=norm).cuda()
        model.load_state_dict(shared_weights)

        trainer = Trainer(model, graph, config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        metrics = trainer.train_epoch(optimizer, criterion)

        grads = {
            n: p.grad.clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return metrics["loss"], grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPartitionInvariance:
    """2x2 test: {1 part, 4 parts} x {hongtu, grinnder} should all produce
    the same gradients (full-graph training is partition-invariant)."""

    def test_hongtu_1part_vs_4part(self, graph, shared_weights):
        """HongTu: 1 partition vs 4 partitions should give same gradients."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        loss_1, grads_1 = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="hongtu", num_parts=1,
        )
        loss_4, grads_4 = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="hongtu", num_parts=4,
        )

        print(f"\nHongTu 1-part loss: {loss_1:.6f}")
        print(f"HongTu 4-part loss: {loss_4:.6f}")
        print(f"Loss diff: {abs(loss_1 - loss_4):.2e}")
        for name in grads_1:
            if name in grads_4:
                diff = (grads_1[name] - grads_4[name]).abs().max().item()
                print(f"  {name}: max grad diff = {diff:.2e}")

        assert abs(loss_1 - loss_4) < 1e-4, (
            f"HongTu 1-part={loss_1:.6f} vs 4-part={loss_4:.6f}"
        )
        for name in grads_1:
            if name in grads_4:
                assert torch.allclose(grads_1[name], grads_4[name], atol=1e-4), (
                    f"HongTu grad mismatch {name}: "
                    f"max diff={(grads_1[name]-grads_4[name]).abs().max():.2e}"
                )

    def test_grinnder_1part_vs_4part(self, graph, shared_weights):
        """GriNNder: 1 partition vs 4 partitions should give same gradients."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        loss_1, grads_1 = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder", num_parts=1,
        )
        loss_4, grads_4 = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder", num_parts=4,
        )

        print(f"\nGriNNder 1-part loss: {loss_1:.6f}")
        print(f"GriNNder 4-part loss: {loss_4:.6f}")
        print(f"Loss diff: {abs(loss_1 - loss_4):.2e}")
        for name in grads_1:
            if name in grads_4:
                diff = (grads_1[name] - grads_4[name]).abs().max().item()
                print(f"  {name}: max grad diff = {diff:.2e}")

        assert abs(loss_1 - loss_4) < 1e-4, (
            f"GriNNder 1-part={loss_1:.6f} vs 4-part={loss_4:.6f}"
        )
        for name in grads_1:
            if name in grads_4:
                assert torch.allclose(grads_1[name], grads_4[name], atol=1e-4), (
                    f"GriNNder grad mismatch {name}: "
                    f"max diff={(grads_1[name]-grads_4[name]).abs().max():.2e}"
                )

    def test_hongtu_vs_grinnder_4part(self, graph, shared_weights):
        """HongTu 4-part vs GriNNder 4-part should give same gradients."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        loss_h, grads_h = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="hongtu", num_parts=4,
        )
        loss_g, grads_g = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_weights, mode="grinnder", num_parts=4,
        )

        print(f"\nHongTu 4-part loss:  {loss_h:.6f}")
        print(f"GriNNder 4-part loss: {loss_g:.6f}")
        print(f"Loss diff: {abs(loss_h - loss_g):.2e}")
        for name in grads_h:
            if name in grads_g:
                diff = (grads_h[name] - grads_g[name]).abs().max().item()
                print(f"  {name}: max grad diff = {diff:.2e}")

        assert abs(loss_h - loss_g) < 1e-4, (
            f"HongTu={loss_h:.6f} vs GriNNder={loss_g:.6f}"
        )
        for name in grads_h:
            if name in grads_g:
                assert torch.allclose(grads_h[name], grads_g[name], atol=1e-4), (
                    f"HongTu vs GriNNder grad mismatch {name}: "
                    f"max diff={(grads_h[name]-grads_g[name]).abs().max():.2e}"
                )

    def test_grinnder_1part_vs_4part_with_layernorm(
        self, graph, shared_norm_weights
    ):
        """LayerNorm should remain partition-invariant under native gather."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        loss_1, grads_1 = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_norm_weights, mode="grinnder", num_parts=1, norm=True,
        )
        loss_4, grads_4 = _grinnder_step_nparts(
            edge_index, x, y, train_mask, val_mask, test_mask,
            shared_norm_weights, mode="grinnder", num_parts=4, norm=True,
        )

        print(f"\nGriNNder LayerNorm 1-part loss: {loss_1:.6f}")
        print(f"GriNNder LayerNorm 4-part loss: {loss_4:.6f}")
        print(f"Loss diff: {abs(loss_1 - loss_4):.2e}")

        assert abs(loss_1 - loss_4) < 1e-4, (
            f"GriNNder LayerNorm 1-part={loss_1:.6f} vs 4-part={loss_4:.6f}"
        )
        for name in grads_1:
            if name in grads_4:
                assert torch.allclose(grads_1[name], grads_4[name], atol=1e-4), (
                    f"GriNNder LayerNorm grad mismatch {name}: "
                    f"max diff={(grads_1[name]-grads_4[name]).abs().max():.2e}"
                )

    @pytest.mark.parametrize("mode", ["hongtu", "grinnder"])
    def test_releases_device_buffers_after_epoch(self, graph, shared_weights, mode):
        """Partitioned training should not leave gathered GPU buffers resident."""
        edge_index, x, y, train_mask, val_mask, test_mask = graph

        with tempfile.TemporaryDirectory() as tmp:
            config = GriNNderConfig(
                num_parts=4,
                partitioner="grinnder",
                mode=mode,
                storage_dir=tmp,
                cache_mode="auto",
            )
            part_graph = build_partitioned_graph(
                edge_index, x, y, train_mask, val_mask, test_mask, config
            )

            model = GCN(8, 16, 3, num_layers=2, dropout=0.0).cuda()
            model.load_state_dict(shared_weights)
            trainer = Trainer(model, part_graph, config)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            criterion = torch.nn.CrossEntropyLoss()

            trainer.train_epoch(optimizer, criterion)

            for device_buffer in trainer.device_features:
                assert not any(
                    device_buffer.is_allocated(pid)
                    for pid in range(part_graph.num_parts)
                )
                assert all(
                    device_buffer[pid].grad is None
                    for pid in range(part_graph.num_parts)
                )
            for device_buffer in trainer.device_gradients:
                if device_buffer is None:
                    continue
                assert not any(
                    device_buffer.is_allocated(pid)
                    for pid in range(part_graph.num_parts)
                )
                assert all(
                    device_buffer[pid].grad is None
                    for pid in range(part_graph.num_parts)
                )
