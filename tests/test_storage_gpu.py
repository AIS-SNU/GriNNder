"""Tests for StorageTensor GPU paths and StorageBackend GPU I/O."""

import tempfile

import torch
import pytest

kvikio = pytest.importorskip("kvikio")
from grinnder.storage.backend import StorageBackend
from grinnder.storage.tensor import StorageTensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestStorageTensorGPU:
    def test_fill_from_gpu(self):
        """D2H: GPU tensor -> StorageTensor host buffer."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = StorageBackend(tmp)
            st = StorageTensor((5, 3), torch.float32, backend, "test_fill")

            gpu_data = torch.randn(5, 3, device="cuda")
            st.fill_from_gpu(gpu_data)

            assert st.location == "host"
            assert torch.allclose(st.host_data, gpu_data.cpu())

    def test_to_gpu(self):
        """H2D: StorageTensor host buffer -> GPU tensor."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = StorageBackend(tmp)
            st = StorageTensor((5, 3), torch.float32, backend, "test_togpu")
            st.host_data.fill_(2.5)

            gpu_buf = torch.empty(5, 3, device="cuda")
            st.to_gpu(gpu_buf)
            torch.cuda.synchronize()

            assert torch.allclose(gpu_buf.cpu(), torch.full((5, 3), 2.5))

    def test_full_lifecycle_with_gpu(self):
        """Full cycle: fill from GPU -> to storage -> to host -> to GPU."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = StorageBackend(tmp)
            st = StorageTensor((10, 4), torch.float32, backend, "lifecycle")

            # Fill from GPU
            original = torch.randn(10, 4, device="cuda")
            st.fill_from_gpu(original)

            # To storage
            st.to_storage(async_=False)
            assert st.location == "storage"

            # Back to host
            st.to_host(async_=False)
            st.finish_to_host()
            assert st.location == "host"

            # To GPU
            gpu_out = torch.empty(10, 4, device="cuda")
            st.to_gpu(gpu_out)
            torch.cuda.synchronize()

            assert torch.allclose(gpu_out.cpu(), original.cpu())
