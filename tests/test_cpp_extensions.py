"""Tests for C++ extensions: async transfers, gather/scatter, subgraph, io_uring."""

import tempfile
import os

import torch
import pytest


@pytest.fixture
def cpp():
    """Load C++ extension, skip if not available."""
    try:
        import grinnder._C as _C
        return _C
    except ImportError:
        pytest.skip("C++ extension not built")


class TestIoUringEngine:
    def test_creation(self, cpp):
        engine = cpp.IoUringEngine(64)
        assert engine.pending() == 0

    def test_has_io_uring(self, cpp):
        engine = cpp.IoUringEngine(64)
        # Should be True if liburing was linked, False for POSIX fallback
        assert isinstance(engine.has_io_uring(), bool)

    def test_write_read_roundtrip(self, cpp):
        engine = cpp.IoUringEngine(64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            path = f.name

        try:
            src = torch.randn(100, 16)
            handle = engine.submit_write(path, src, 0, src.numel() * src.element_size())
            engine.wait(handle)

            dst = torch.empty_like(src)
            handle = engine.submit_read(path, dst, 0, dst.numel() * dst.element_size())
            engine.wait(handle)

            assert torch.allclose(src, dst), "io_uring write/read roundtrip mismatch"
        finally:
            os.unlink(path)

    def test_poll(self, cpp):
        engine = cpp.IoUringEngine(64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            path = f.name

        try:
            src = torch.randn(10)
            handle = engine.submit_write(path, src, 0, src.numel() * src.element_size())
            # Poll until done (non-blocking check)
            engine.wait(handle)
            # After wait, pending should be 0
            assert engine.pending() == 0
        finally:
            os.unlink(path)


class TestBuildSubgraph:
    def test_simple_graph(self, cpp):
        # Graph: 0->1, 0->2, 1->2, 2->3
        rowptr = torch.tensor([0, 2, 3, 4, 4], dtype=torch.long)
        col = torch.tensor([1, 2, 2, 3], dtype=torch.long)
        seed = torch.tensor([0, 1], dtype=torch.long)  # dest nodes

        out_rowptr, out_col, out_value, n_id = cpp.build_subgraph(
            rowptr, col, None, seed, True  # bipartite
        )

        # n_id should contain seed nodes + boundary nodes
        assert n_id[:2].tolist() == [0, 1]
        # Boundary nodes: 2 (neighbor of both 0 and 1)
        assert 2 in n_id[2:].tolist()
        # rowptr should have len(seed)+1 entries (bipartite)
        assert out_rowptr.shape[0] == 3

    def test_with_values(self, cpp):
        rowptr = torch.tensor([0, 2, 3, 4, 4], dtype=torch.long)
        col = torch.tensor([1, 2, 2, 3], dtype=torch.long)
        value = torch.tensor([0.5, 0.3, 0.7, 0.1], dtype=torch.float)
        seed = torch.tensor([0], dtype=torch.long)

        out_rowptr, out_col, out_value, n_id = cpp.build_subgraph(
            rowptr, col, value, seed, True
        )

        assert out_value is not None
        assert out_value.shape[0] == out_col.shape[0]

    def test_non_bipartite(self, cpp):
        rowptr = torch.tensor([0, 2, 3, 3], dtype=torch.long)
        col = torch.tensor([1, 2, 2], dtype=torch.long)
        seed = torch.tensor([0], dtype=torch.long)

        out_rowptr, out_col, out_value, n_id = cpp.build_subgraph(
            rowptr, col, None, seed, False  # non-bipartite
        )

        # rowptr should have len(n_id)+1 entries
        assert out_rowptr.shape[0] == n_id.shape[0] + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAsyncTransfers:
    def test_h2d_copy(self, cpp):
        src = torch.randn(10, 4, pin_memory=True)
        dst = torch.empty(10, 4, device="cuda")

        # Use non-blocking copy directly to verify CUDA path works
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
        assert torch.allclose(src, dst.cpu()), "basic non_blocking copy failed"

        # Now test C++ h2d: must set stream explicitly for getCurrentCUDAStream
        dst.zero_()
        torch.cuda.set_stream(stream)
        cpp.h2d_copy_async(src, dst)
        cpp.h2d_synchronize()
        torch.cuda.synchronize()  # ensure all CUDA ops complete
        torch.cuda.set_stream(torch.cuda.default_stream())

        assert torch.allclose(src, dst.cpu())

    def test_d2h_copy(self, cpp):
        src = torch.randn(10, 4, device="cuda")
        dst = torch.empty(10, 4, pin_memory=True)
        stream = torch.cuda.Stream()

        torch.cuda.set_stream(stream)
        cpp.d2h_copy_async(src, dst)
        cpp.d2h_synchronize()
        torch.cuda.synchronize()  # device-level fence
        torch.cuda.set_stream(torch.cuda.default_stream())

        assert torch.allclose(src.cpu(), dst)

    def test_gather_partitions(self, cpp):
        # 3 partitions: sizes 4, 3, 5. Gather for pid=0.
        # Boundaries: pid=0 needs indices [0,2] from pid=1 and [3] from pid=2
        p0 = torch.arange(8, dtype=torch.float).reshape(4, 2).pin_memory()
        p1 = torch.arange(8, 14, dtype=torch.float).reshape(3, 2).pin_memory()
        p2 = torch.arange(14, 24, dtype=torch.float).reshape(5, 2).pin_memory()
        srcs = [p0, p1, p2]

        boundaries = [
            torch.empty(0, dtype=torch.long),  # pid=0 (self)
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        ]

        total = 4 + 2 + 1
        dst = torch.empty(total, 2, device="cuda")
        stream = torch.cuda.Stream()

        torch.cuda.set_stream(stream)
        cpp.gather_partitions(0, srcs, dst, boundaries)
        cpp.h2d_synchronize()
        torch.cuda.synchronize()
        torch.cuda.set_stream(torch.cuda.default_stream())

        result = dst.cpu()
        assert torch.allclose(result[:4], p0)
        assert torch.allclose(result[4:6], p1.index_select(0, torch.tensor([0, 2])))
        assert torch.allclose(result[6:7], p2.index_select(0, torch.tensor([3])))

    def test_scatter_partitions(self, cpp):
        # Scatter from GPU back to 3 host partitions with accumulation
        p0 = torch.zeros(4, 2, pin_memory=True)
        p1 = torch.zeros(3, 2, pin_memory=True)
        p2 = torch.zeros(5, 2, pin_memory=True)
        dsts = [p0, p1, p2]

        boundaries = [
            torch.empty(0, dtype=torch.long),
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        ]

        # GPU source: [4 intra + 2 boundary_p1 + 1 boundary_p2]
        src = torch.ones(7, 2, device="cuda")
        stream = torch.cuda.Stream()

        torch.cuda.set_stream(stream)
        cpp.scatter_partitions(0, src, dsts, boundaries)
        cpp.d2h_synchronize()
        torch.cuda.synchronize()  # device-level fence
        torch.cuda.set_stream(torch.cuda.default_stream())

        # p0 should have 1s (D2H overwrite + accumulate from zero = 1.0)
        assert torch.allclose(p0, torch.ones(4, 2), atol=1e-5), f"p0 mismatch: {p0}"
        # p1[0] and p1[2] should have accumulated boundary values
        assert p1[0].sum() > 0, f"p1[0] empty: {p1}"
        assert p1[2].sum() > 0, f"p1[2] empty: {p1}"
        # p2[3] should have accumulated boundary values
        assert p2[3].sum() > 0, f"p2[3] empty: {p2}"
