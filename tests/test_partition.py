"""Tests for build_partitioned_graph end-to-end."""

from concurrent.futures import ThreadPoolExecutor

import torch
import pytest

pyg = pytest.importorskip("torch_geometric")
grdpart = pytest.importorskip("grdpart")

from grinnder.config import GriNNderConfig
from grinnder.data.datasets import compute_gcn_norm
from grinnder.data.partition import (
    _build_partition_subgraph,
    _edge_index_to_csr,
    _load_subgraph_fn,
    build_partitioned_graph,
)


@pytest.fixture
def simple_graph():
    """Small graph: 20 nodes, ~50 edges."""
    num_nodes = 20
    # Create a random connected graph
    edges = set()
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 4, num_nodes)):
            edges.add((i, j))
            edges.add((j, i))
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    edge_index = torch.tensor([row, col], dtype=torch.long)

    x = torch.randn(num_nodes, 8)
    y = torch.randint(0, 3, (num_nodes,))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:14] = True
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[14:17] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[17:] = True

    return edge_index, x, y, train_mask, val_mask, test_mask


class TestBuildPartitionedGraph:
    def test_basic_partitioning(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")

        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        assert graph.num_nodes == 20
        assert graph.num_parts == 4
        assert len(graph.partition_sizes) == 4
        assert sum(graph.partition_sizes) == 20
        assert graph.feat_dim == 8
        assert graph.num_classes == 3

    def test_partition_sizes_match_ptr(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        for pid in range(4):
            expected_size = int(graph.ptr[pid + 1] - graph.ptr[pid])
            assert graph.partition_sizes[pid] == expected_size

    def test_features_reordered(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=2, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        # Features should be reordered by perm
        assert graph.features.shape == x.shape
        # Reordered features should contain the same values (just permuted)
        orig_sorted = x[graph.perm].sort(dim=0).values
        reord_sorted = graph.features.sort(dim=0).values
        assert torch.allclose(orig_sorted, reord_sorted)

    def test_masks_preserved(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        # Total mask counts should be preserved
        assert graph.train_mask.sum() == train_mask.sum()
        assert graph.val_mask.sum() == val_mask.sum()
        assert graph.test_mask.sum() == test_mask.sum()

    def test_boundaries_structure(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        assert len(graph.boundaries) == 4
        for pid in range(4):
            assert len(graph.boundaries[pid]) == 4
            # Self-boundary should be None
            assert graph.boundaries[pid][pid] is None

    def test_adj_bipartite(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        for pid in range(4):
            rowptr, col, value = graph.adj_csr[pid]
            batch_size = graph.partition_sizes[pid]
            expanded = graph.expanded_sizes[pid]

            # rowptr should have batch_size + 1 entries (bipartite)
            assert rowptr.shape[0] == batch_size + 1
            # col values should be in [0, expanded)
            if col.numel() > 0:
                assert col.max() < expanded
                assert col.min() >= 0

    def test_expanded_sizes_ge_partition_sizes(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        for pid in range(4):
            # Expanded size >= partition size (includes boundary nodes)
            assert graph.expanded_sizes[pid] >= graph.partition_sizes[pid]

    def test_partition_accessors(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )

        for pid in range(4):
            feat = graph.partition_features(pid)
            assert feat.shape[0] == graph.partition_sizes[pid]
            assert feat.shape[1] == 8

            labels = graph.partition_labels(pid)
            assert labels.shape[0] == graph.partition_sizes[pid]

    def test_grinnder_partitioner(self, simple_graph):
        """Test with the actual grinnder partitioner (power-of-2 num_parts)."""
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")
        graph = build_partitioned_graph(
            edge_index, x, y, train_mask, val_mask, test_mask, config
        )
        assert graph.num_parts == 4
        assert sum(graph.partition_sizes) == 20

    def test_partition_cache_round_trip(self, simple_graph, tmp_path):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        cache_path = tmp_path / "graph_cache.pt"
        config = GriNNderConfig(num_parts=4, partitioner="grinnder")

        graph = build_partitioned_graph(
            edge_index,
            x,
            y,
            train_mask,
            val_mask,
            test_mask,
            config,
            cache_path=cache_path,
        )
        assert cache_path.exists()

        graph_cached = build_partitioned_graph(
            edge_index,
            x,
            y,
            train_mask,
            val_mask,
            test_mask,
            config,
            cache_path=cache_path,
        )

        assert graph_cached.partition_sizes == graph.partition_sizes
        assert graph_cached.expanded_sizes == graph.expanded_sizes
        assert torch.equal(graph_cached.perm, graph.perm)
        assert torch.equal(graph_cached.ptr, graph.ptr)
        for pid in range(config.num_parts):
            for cached, original in zip(graph_cached.adj_csr[pid], graph.adj_csr[pid]):
                if cached is None or original is None:
                    assert cached is original
                else:
                    assert torch.equal(cached, original)

    def test_parallel_preprocess_matches_serial(self, simple_graph):
        edge_index, x, y, train_mask, val_mask, test_mask = simple_graph
        del x, y, train_mask, val_mask, test_mask
        num_nodes = 20
        edge_weight = compute_gcn_norm(edge_index, num_nodes, add_self_loops=True)
        from torch_geometric.utils import add_self_loops

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        rowptr, col, sort_perm = _edge_index_to_csr(edge_index, num_nodes)
        edge_weight = edge_weight[sort_perm]
        ptr = torch.tensor([0, 5, 10, 15, 20], dtype=torch.long)
        build_subgraph = _load_subgraph_fn()
        args = (ptr, rowptr, col, edge_weight, 4, build_subgraph)

        serial = [
            _build_partition_subgraph(pid, *args)
            for pid in range(4)
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel = list(
                executor.map(
                    lambda pid: _build_partition_subgraph(pid, *args),
                    range(4),
                )
            )

        for serial_result, parallel_result in zip(serial, parallel):
            assert serial_result[0] == parallel_result[0]
            assert serial_result[3:] == parallel_result[3:]
            for serial_tensor, parallel_tensor in zip(
                serial_result[1], parallel_result[1]
            ):
                if serial_tensor is None or parallel_tensor is None:
                    assert serial_tensor is parallel_tensor
                else:
                    assert torch.equal(serial_tensor, parallel_tensor)
            for serial_boundary, parallel_boundary in zip(
                serial_result[2], parallel_result[2]
            ):
                if serial_boundary is None or parallel_boundary is None:
                    assert serial_boundary is parallel_boundary
                else:
                    assert torch.equal(serial_boundary, parallel_boundary)
