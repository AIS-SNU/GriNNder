"""Tests for autograd: GradOffload and ScatteredCheckpoint."""

import torch
import pytest

from grinnder.buffer.host import HostBuffer
from grinnder.buffer.device import DeviceBuffer
from grinnder.autograd.grad_offload import GradOffload
from grinnder.autograd.checkpoint import ScatteredCheckpoint, HongtuCheckpoint
from torch.autograd.graph import saved_tensors_hooks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGradOffload:
    def test_forward_is_identity(self):
        """GradOffload.forward should return input unchanged."""
        x = torch.randn(10, 4, device="cuda", requires_grad=True)
        host_grads = HostBuffer(num_parts=2, part_sizes=[10, 8], embedding_dim=4)
        boundaries = [None, torch.tensor([0, 1, 2], dtype=torch.long)]
        stream = torch.cuda.Stream()

        out = GradOffload.apply(
            x, 1, 0, host_grads, boundaries, stream, torch.cuda.current_stream()
        )
        assert torch.allclose(out, x)

    def test_backward_scatters_gradients(self):
        """GradOffload.backward should scatter gradients to host buffer."""
        # Setup: 2 partitions, partition 0 has 5 intra + 3 boundary from partition 1
        host_grads = HostBuffer(num_parts=2, part_sizes=[5, 8], embedding_dim=4)
        host_grads.initialize_zeros()

        boundaries = [None, torch.tensor([0, 2, 4], dtype=torch.long)]
        d2h_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.current_stream()

        # Create input that requires grad
        x = torch.randn(8, 4, device="cuda", requires_grad=True)  # 5 intra + 3 boundary

        # Forward: identity
        out = GradOffload.apply(
            x, 1, 0, host_grads, boundaries, d2h_stream, compute_stream
        )

        # Simulate a loss and backward
        loss = out.sum()
        loss.backward()

        # Wait for scatter to complete
        host_grads.d2h_synchronize(d2h_stream)

        # host_grads[0] should have non-zero values (accumulated from scatter)
        assert host_grads[0].abs().sum() > 0, "Gradients were not scattered to host"
        assert x.grad is None

    def test_layer0_skips_scatter(self):
        """GradOffload at layer 0 should not scatter (no input gradient)."""
        host_grads = HostBuffer(num_parts=2, part_sizes=[5, 5], embedding_dim=4)
        host_grads.initialize_zeros()
        boundaries = [None, torch.empty(0, dtype=torch.long)]
        stream = torch.cuda.Stream()

        x = torch.randn(5, 4, device="cuda", requires_grad=True)

        # layer_id=0: backward should stop without scatter or x.grad materialization
        out = GradOffload.apply(x, 0, 0, host_grads, boundaries, stream, torch.cuda.current_stream())
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()

        # host_grads should remain zero (no scatter at layer 0)
        assert host_grads[0].abs().sum() == 0
        assert x.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestScatteredCheckpoint:
    def test_scattered_checkpoint_is_saved_tensors_hooks(self):
        """ScatteredCheckpoint inherits from saved_tensors_hooks."""
        device_buf = DeviceBuffer(num_parts=1, part_sizes=[5], embedding_dim=4)
        device_buf.allocate(0)
        h2d_stream = torch.cuda.Stream()

        ctx = ScatteredCheckpoint(0, device_buf, h2d_stream)
        assert isinstance(ctx, saved_tensors_hooks)

    def test_hongtu_checkpoint(self):
        """HongtuCheckpoint should save to CPU and restore."""
        from torch.utils.checkpoint import checkpoint

        device_buf = DeviceBuffer(num_parts=1, part_sizes=[5], embedding_dim=4)
        device_buf.allocate(0)

        def fn(x):
            return x * 3

        x = torch.randn(5, 4, device="cuda", requires_grad=True)

        with HongtuCheckpoint(0, device_buf):
            y = checkpoint(fn, x, use_reentrant=True)

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.full((5, 4), 3.0, device="cuda"))
