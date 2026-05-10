"""Tests for PartitionCache: auto-detect, LRU modes, eviction, hit rates."""

import tempfile

import torch
import pytest

from grinnder.buffer.host import HostBuffer
from grinnder.cache.partition_cache import PartitionCache


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class FakeBackend:
    def __init__(self):
        self.store = {}

    def host_write(self, tensor, file_id, async_=True):
        self.store[file_id] = tensor.detach().clone()
        return file_id

    def host_read(self, file_id, tensor, async_=True):
        tensor.copy_(self.store[file_id])
        return file_id

    def wait(self, handle):
        return None

    def exists(self, file_id):
        return file_id in self.store


@pytest.fixture
def backend():
    return FakeBackend()


def _gb(nbytes):
    return nbytes / float(1024**3)


def _make_cache(
    num_parts,
    num_layers,
    dim,
    num_nodes,
    backend,
    mode,
    budget_gb=100,
    fixed_resident_bytes=0,
    safety_margin_bytes=0,
    dependency_sets=None,
    part_sizes=None,
):
    """Helper: create host buffers + gradient buffers + cache."""
    layer_dims = [dim] * (num_layers + 1)
    if part_sizes is None:
        part_sizes = [num_nodes // num_parts] * num_parts

    host_buffers = []
    for l in range(num_layers + 1):
        buf = HostBuffer(num_parts, part_sizes, dim, backend=backend, file_prefix=f"feat_l{l}")
        host_buffers.append(buf)

    grad_buffers = [None]  # layer 0
    for l in range(1, num_layers):
        buf = HostBuffer(num_parts, part_sizes, dim, backend=backend, file_prefix=f"grad_l{l}")
        grad_buffers.append(buf)

    cache = PartitionCache(
        num_parts=num_parts,
        num_layers=num_layers,
        layer_dims=layer_dims,
        num_nodes=num_nodes,
        host_buffers=host_buffers,
        grad_buffers=grad_buffers,
        backend=backend,
        mode=mode,
        host_memory_budget_gb=budget_gb,
        fixed_resident_bytes=fixed_resident_bytes,
        safety_margin_bytes=safety_margin_bytes,
        dependency_sets=dependency_sets,
    )
    return cache, host_buffers, grad_buffers


class TestAutoDetect:
    def test_lru_layer_when_plenty_of_memory(self, backend):
        # 100 nodes, dim=4, 3 layers -> tiny memory. 100GB budget -> lru_layer.
        cache, _, _ = _make_cache(4, 3, 4, 100, backend, mode="auto", budget_gb=100)
        assert cache.mode == "lru_layer"

    def test_uses_remaining_budget_after_fixed_resident_data(self, backend):
        fixed = 256
        cache, _, _ = _make_cache(
            2,
            2,
            4,
            10,
            backend,
            mode="auto",
            budget_gb=100,
            fixed_resident_bytes=fixed,
            safety_margin_bytes=64,
        )
        plan = cache.memory_plan

        full_budget = fixed + 64 + plan.all_layer_residency_bytes
        cache, _, _ = _make_cache(
            2,
            2,
            4,
            10,
            backend,
            mode="auto",
            budget_gb=_gb(full_budget),
            fixed_resident_bytes=fixed,
            safety_margin_bytes=64,
        )
        assert cache.mode == "lru_layer"

        max_gradient = max(plan.gradient_layer_bytes)
        lru_budget = fixed + 64 + max_gradient + plan.layer_working_set_bytes
        cache, _, _ = _make_cache(
            2,
            2,
            4,
            10,
            backend,
            mode="auto",
            budget_gb=_gb(lru_budget),
            fixed_resident_bytes=fixed,
            safety_margin_bytes=64,
        )
        assert cache.mode == "lru_layer"

        cache, _, _ = _make_cache(
            2,
            2,
            4,
            10,
            backend,
            mode="auto",
            budget_gb=_gb(lru_budget - 1),
            fixed_resident_bytes=fixed,
            safety_margin_bytes=64,
        )
        assert cache.mode == "partition_lru"

    def test_explicit_mode(self, backend):
        cache, _, _ = _make_cache(4, 3, 4, 100, backend, mode="partition_lru")
        assert cache.mode == "partition_lru"

    def test_partition_lru_uses_full_remaining_activation_budget(self, backend):
        cache, _, _ = _make_cache(
            2,
            2,
            4,
            10,
            backend,
            mode="partition_lru",
            budget_gb=1,
        )
        assert cache.activation_cache_budget_bytes == cache.remaining_cache_bytes

    def test_full_layer_is_not_supported(self, backend):
        with pytest.raises(ValueError, match="Unsupported cache mode"):
            _make_cache(4, 3, 4, 100, backend, mode="full_layer")


class TestLruLayerMode:
    def test_load_and_evict(self, backend):
        cache, bufs, _ = _make_cache(4, 3, 8, 100, backend, mode="lru_layer", budget_gb=0.0001)

        # Pre-write layer 0 data to storage so storage_to_cpu can load it
        for pid in range(4):
            bufs[0][pid].fill_(1.0)
            bufs[0].cpu_to_storage(pid)

        # First access: miss (loads from storage)
        cache.ensure_in_host(0)
        assert cache._misses >= 1

    def test_hit_on_second_access(self, backend):
        cache, bufs, _ = _make_cache(4, 3, 8, 100, backend, mode="lru_layer")

        for pid in range(4):
            bufs[0][pid].fill_(1.0)
            bufs[0].cpu_to_storage(pid)
            bufs[0].release(pid)

        cache.ensure_in_host(0)
        hits_before = cache._hits
        cache.ensure_in_host(0)
        assert cache._hits == hits_before + 1  # second access is a hit

    def test_keeps_new_layers_when_budget_allows(self, backend):
        cache, bufs, _ = _make_cache(
            2,
            2,
            1,
            20,
            backend,
            mode="lru_layer",
            budget_gb=100,
            part_sizes=[10, 10],
        )

        for pid in range(2):
            bufs[0][pid].fill_(1.0)
            bufs[0].cpu_to_storage(pid)
            bufs[0].release(pid)

        cache.ensure_in_host(0)
        for pid in range(2):
            bufs[1][pid].fill_(2.0)
        cache.on_layer_complete(0)

        assert all(bufs[0].is_allocated(pid) for pid in range(2))
        assert all(bufs[1].is_allocated(pid) for pid in range(2))
        assert list(cache._cached_layers.keys()) == [0, 1]

    def test_eviction_writes_new_layer_to_storage(self, backend):
        layer_bytes = 2 * 10 * 1 * 4
        cache, bufs, _ = _make_cache(
            2,
            2,
            1,
            20,
            backend,
            mode="lru_layer",
            budget_gb=_gb(layer_bytes * 2),
            part_sizes=[10, 10],
        )

        for pid in range(2):
            bufs[0][pid].fill_(1.0)
            bufs[0].cpu_to_storage(pid)
            bufs[0].release(pid)

        cache.ensure_in_host(0)
        for pid in range(2):
            bufs[1][pid].fill_(2.0)
        cache.on_layer_complete(0)
        for pid in range(2):
            bufs[2][pid].fill_(3.0)
        cache.on_layer_complete(1)

        assert all(backend.exists(f"feat_l1_p{pid}") for pid in range(2))
        assert all(not bufs[1].is_allocated(pid) for pid in range(2))
        assert all(bufs[2].is_allocated(pid) for pid in range(2))
        assert list(cache._cached_layers.keys()) == [2]


class TestPartitionLruMode:
    def test_ensure_partition_in_host(self, backend):
        cache, bufs, _ = _make_cache(4, 3, 8, 100, backend, mode="partition_lru", budget_gb=100)

        # Pre-write partition data
        for pid in range(4):
            bufs[0][pid].fill_(float(pid))
            bufs[0].cpu_to_storage(pid)

        cache.ensure_partition_in_host(0, 0)
        cache.ensure_partition_in_host(0, 1)
        assert cache._hits == 0  # both are misses (first access)
        assert cache._misses == 2

        # Second access should hit
        cache.ensure_partition_in_host(0, 0)
        assert cache._hits == 1

    def test_dependency_set_replacement_keeps_reused_partition(self, backend):
        cache, bufs, _ = _make_cache(
            3,
            1,
            1,
            30,
            backend,
            mode="partition_lru",
            budget_gb=_gb(80),
            part_sizes=[10, 10, 10],
        )

        for pid in range(3):
            bufs[0][pid].fill_(float(pid))
            bufs[0].cpu_to_storage(pid)
        bufs[0].release_all()

        cache.ensure_dependencies_in_host(0, 0, dependencies={0, 2})
        assert bufs[0].is_allocated(0)
        assert not bufs[0].is_allocated(1)
        assert bufs[0].is_allocated(2)

        cache.ensure_dependencies_in_host(0, 1, dependencies={1, 2})
        assert not bufs[0].is_allocated(0)
        assert bufs[0].is_allocated(1)
        assert bufs[0].is_allocated(2)
        assert set(cache._cached_partitions.keys()) == {(0, 1), (0, 2)}
        assert cache._hits == 1
        assert cache._misses == 3


class TestBackwardGradientFlush:
    def test_on_backward_layer_complete(self, backend):
        cache, bufs, grad_bufs = _make_cache(4, 3, 8, 100, backend, mode="lru_layer")

        # Fill gradient buffer with data
        grad_bufs[1][0].fill_(3.14)

        # Flush to storage
        cache.on_backward_layer_complete(1)

        # Verify it was written
        assert backend.exists("grad_l1_p0")

    def test_partition_lru_flushes_only_resident_gradient_partitions(self, backend):
        cache, _, grad_bufs = _make_cache(
            4,
            3,
            8,
            100,
            backend,
            mode="partition_lru",
        )

        grad_bufs[1].initialize_zeros(lazy=True)
        grad_bufs[1][2].fill_(2.0)

        cache.on_backward_layer_complete(1)

        assert backend.exists("grad_l1_p2")
        assert not backend.exists("grad_l1_p0")
        assert not any(grad_bufs[1].is_allocated(pid) for pid in range(4))


class TestReset:
    def test_reset_clears_state(self, backend):
        cache, bufs, _ = _make_cache(4, 3, 4, 100, backend, mode="lru_layer")
        for pid in range(4):
            bufs[0][pid].fill_(1.0)
            bufs[0].cpu_to_storage(pid)
            bufs[0].release(pid)

        cache.ensure_in_host(0)  # miss
        cache.ensure_in_host(0)  # hit

        cache.reset()
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache.hit_rate == 0.0
