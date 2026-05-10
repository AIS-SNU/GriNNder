"""Tests for HostBuffer: scatter, fill, upload, bypass, storage round-trips."""

import tempfile

import torch
import pytest

from grinnder.buffer.host import HostBuffer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestHostBufferFillUpload:
    def test_async_fill_and_upload(self):
        """D2H fill followed by H2D upload should round-trip correctly."""
        buf = HostBuffer(num_parts=2, part_sizes=[5, 3], embedding_dim=4)
        buf.initialize_zeros()

        gpu_data = torch.randn(5, 4, device="cuda")
        stream = torch.cuda.Stream()

        # D2H fill
        torch.cuda.set_stream(stream)
        buf.async_fill(0, gpu_data, stream)
        buf.d2h_synchronize(stream)
        torch.cuda.synchronize()
        torch.cuda.set_stream(torch.cuda.default_stream())

        assert torch.allclose(buf[0], gpu_data.cpu())

        # H2D upload
        gpu_target = torch.empty(5, 4, device="cuda")
        torch.cuda.set_stream(stream)
        buf.async_upload(0, gpu_target, stream)
        buf.h2d_synchronize(stream)
        torch.cuda.synchronize()
        torch.cuda.set_stream(torch.cuda.default_stream())

        assert torch.allclose(gpu_target.cpu(), gpu_data.cpu())

    def test_async_scatter(self):
        """Scatter from GPU to multiple host partitions with accumulation."""
        buf = HostBuffer(num_parts=3, part_sizes=[4, 3, 5], embedding_dim=2)
        buf.initialize_zeros()

        # Set initial values so accumulation is visible
        buf[0].fill_(1.0)

        # GPU source: [4 intra + 2 from p1 + 1 from p2]
        boundaries = [None, torch.tensor([0, 2]), torch.tensor([3])]
        gpu_src = torch.ones(7, 2, device="cuda") * 2.0
        stream = torch.cuda.Stream()

        torch.cuda.set_stream(stream)
        buf.async_scatter(0, gpu_src, boundaries, stream)
        buf.d2h_synchronize(stream)
        torch.cuda.synchronize()  # device-level fence
        torch.cuda.set_stream(torch.cuda.default_stream())

        # p0 should be 1.0 (original) + 2.0 (scatter) = 3.0
        assert torch.allclose(buf[0], torch.full((4, 2), 3.0), atol=1e-5), f"Expected 3.0, got {buf[0]}"


@pytest.mark.skipif(
    not pytest.importorskip("kvikio", reason="kvikio required"),
    reason="kvikio required"
)
class TestHostBufferStorage:
    def test_cpu_to_storage_and_back(self):
        from grinnder.storage.backend import StorageBackend
        with tempfile.TemporaryDirectory() as tmp:
            backend = StorageBackend(tmp)
            buf = HostBuffer(3, [4, 3, 5], 2, backend=backend, file_prefix="test")

            buf[0].fill_(3.14)
            buf[1].fill_(2.71)

            buf.cpu_to_storage(pid=0)
            buf.cpu_to_storage(pid=1)

            # Zero out and reload
            buf[0].zero_()
            buf[1].zero_()
            buf.storage_to_cpu(pid=0)
            buf.storage_to_cpu(pid=1)

            assert torch.allclose(buf[0], torch.full((4, 2), 3.14))
            assert torch.allclose(buf[1], torch.full((3, 2), 2.71))

    def test_storage_roundtrip_after_zero(self):
        """After zeroing, storage_to_cpu should restore original data."""
        from grinnder.storage.backend import StorageBackend
        with tempfile.TemporaryDirectory() as tmp:
            backend = StorageBackend(tmp)
            buf = HostBuffer(2, [4, 3], 2, backend=backend, file_prefix="evict")

            buf[0].fill_(5.0)
            buf.cpu_to_storage(pid=0)

            # Zero out (simulates data loss, not resize_(0))
            buf[0].zero_()
            assert buf[0].abs().sum() == 0

            # Reload from storage should restore
            buf.storage_to_cpu(pid=0)
            assert buf[0].shape == (4, 2)
            assert torch.allclose(buf[0], torch.full((4, 2), 5.0))
