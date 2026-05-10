"""Tests for HostBuffer and DeviceBuffer."""

import torch
import pytest

from grinnder.buffer.host import HostBuffer
from grinnder.buffer.device import DeviceBuffer


class TestHostBuffer:
    def test_creation(self):
        buf = HostBuffer(num_parts=3, part_sizes=[10, 15, 12], embedding_dim=8)
        assert len(buf) == 3
        assert buf[0].shape == (10, 8)
        assert buf[1].shape == (15, 8)
        assert not buf[0].is_pinned()

    def test_lazy_allocation_and_release(self):
        buf = HostBuffer(num_parts=2, part_sizes=[5, 7], embedding_dim=4)
        assert not buf.is_allocated(0)
        assert not buf.is_allocated(1)
        assert buf.resident_bytes == 0
        assert buf._tensors[0].untyped_storage().size() == 0
        assert tuple(buf._tensors[0].shape) == (0, 4)

        t = buf[0]
        assert t.shape == (5, 4)
        assert buf.is_allocated(0)
        assert buf.resident_bytes == buf.partition_nbytes(0)

        buf.release(0)
        assert not buf.is_allocated(0)
        assert buf.resident_bytes == 0

    def test_initialize_zeros(self):
        buf = HostBuffer(num_parts=2, part_sizes=[5, 5], embedding_dim=4)
        buf[0].fill_(1.0)
        buf.initialize_zeros()
        assert (buf[0] == 0).all()
        assert (buf[1] == 0).all()

    def test_lazy_initialize_zeros(self):
        buf = HostBuffer(num_parts=2, part_sizes=[5, 5], embedding_dim=4)
        buf[0].fill_(1.0)
        buf.initialize_zeros(lazy=True)
        assert not buf.is_allocated(0)
        assert not buf.is_allocated(1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gather_python_fallback(self):
        """Test gather with Python fallback (no C++ extension)."""
        buf = HostBuffer(num_parts=3, part_sizes=[4, 3, 5], embedding_dim=2)
        # Fill with identifiable data
        buf[0].copy_(torch.arange(8).reshape(4, 2).float())
        buf[1].copy_(torch.arange(8, 14).reshape(3, 2).float())
        buf[2].copy_(torch.arange(14, 24).reshape(5, 2).float())

        # Gather for partition 0: intra(4) + boundary from p1(2) + boundary from p2(1)
        boundaries = [None, torch.tensor([0, 2]), torch.tensor([3])]
        total_size = 4 + 2 + 1
        gpu_target = torch.empty(total_size, 2, device="cuda")
        stream = torch.cuda.Stream()

        torch.cuda.set_stream(stream)
        buf.async_gather(0, gpu_target, boundaries, stream)
        buf.h2d_synchronize(stream)
        torch.cuda.synchronize()
        torch.cuda.set_stream(torch.cuda.default_stream())

        result = gpu_target.cpu()
        # Intra: buf[0] = [[0,1],[2,3],[4,5],[6,7]]
        assert torch.allclose(result[:4], buf[0])
        # Boundary from p1 at indices [0,2]: [[8,9],[12,13]]
        assert torch.allclose(result[4:6], buf[1].index_select(0, torch.tensor([0, 2])))
        # Boundary from p2 at index [3]: [[20,21]]
        assert torch.allclose(result[6:7], buf[2].index_select(0, torch.tensor([3])))


class TestDeviceBuffer:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_allocate_release(self):
        buf = DeviceBuffer(num_parts=3, part_sizes=[10, 15, 12], embedding_dim=8)
        assert not buf.is_allocated(0)

        buf.allocate(0)
        assert buf.is_allocated(0)
        assert buf[0].shape == (10, 8)

        buf.release(0)
        assert not buf.is_allocated(0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset(self):
        buf = DeviceBuffer(num_parts=2, part_sizes=[5, 5], embedding_dim=4)
        buf.allocate(0)
        buf.allocate(1)
        buf.reset()
        assert not buf.is_allocated(0)
        assert not buf.is_allocated(1)
