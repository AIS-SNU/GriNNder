"""Storage backend with dual I/O engines.

GPU <-> Storage: kvikio (GPUDirect Storage)
Host <-> Storage: io_uring (bundled liburing 2.8)

Both engines are REQUIRED — no POSIX fallbacks.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

import torch
from torch import Tensor

from grinnder.utils import ensure_dir


class StorageBackend:
    """Manages NVMe file I/O with two dedicated engines.

    - GPU <-> Storage: kvikio (GPUDirect Storage) for bypass writes and
      backward activation loads. Direct GPU-NVMe transfers.
    - Host <-> Storage: io_uring (C++, bundled liburing 2.8) for cache
      loads/flushes and gradient flush. Kernel-level async I/O.

    All files stored under ``storage_dir/{file_id}.pt``.
    """

    def __init__(self, storage_dir: str, queue_depth: int = 64):
        self._storage_dir = ensure_dir(storage_dir)

        # GPU <-> Storage: kvikio required
        try:
            import kvikio
            self._kvikio = kvikio
        except ImportError:
            raise ImportError(
                "kvikio is required for GriNNder's grinnder mode. "
                "Install with: pip install kvikio-cu12"
            )

        # Host <-> Storage: io_uring required (bundled C++ extension)
        try:
            from grinnder._C import IoUringEngine
            self._io_engine = IoUringEngine(queue_depth)
        except ImportError:
            raise ImportError(
                "grinnder._C not found. Build C++ extensions with: "
                "pip install -e . --no-build-isolation"
            )
        if not self._io_engine.has_io_uring():
            raise RuntimeError(
                "io_uring initialization failed. Ensure liburing is available "
                "and the kernel supports io_uring (Linux 5.1+). "
                "The bundled liburing is in third_party/liburing/."
            )

    def _path(self, file_id: str) -> str:
        return os.path.join(self._storage_dir, f"{file_id}.pt")

    # ------------------------------------------------------------------
    # GPU <-> Storage (kvikio / GPUDirect Storage)
    # ------------------------------------------------------------------

    def gpu_write(
        self,
        tensor: Tensor,
        file_id: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Write GPU tensor directly to NVMe via GDS.

        Detaches from autograd graph — bypass writes data only.
        """
        path = self._path(file_id)
        assert tensor.is_cuda, "gpu_write requires a CUDA tensor"

        # Detach and ensure contiguous for GDS __cuda_array_interface__
        t = tensor.detach().contiguous()

        f = self._kvikio.CuFile(path, "w")
        f.write(t, file_offset=0)
        f.close()

    def gpu_read(
        self,
        file_id: str,
        tensor: Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Read from NVMe directly into GPU tensor via GDS."""
        path = self._path(file_id)
        assert tensor.is_cuda, "gpu_read requires a CUDA tensor"
        assert tensor.is_contiguous(), "gpu_read requires contiguous tensor"

        f = self._kvikio.CuFile(path, "r")
        f.read(tensor, file_offset=0)
        f.close()

    # ------------------------------------------------------------------
    # Host <-> Storage (io_uring)
    # ------------------------------------------------------------------

    def host_write(self, tensor: Tensor, file_id: str, async_: bool = True) -> int:
        """Write a CPU tensor to NVMe via io_uring.

        Returns:
            Handle for wait() if async.
        """
        path = self._path(file_id)
        assert not tensor.is_cuda, "host_write requires a CPU tensor"
        assert tensor.is_contiguous(), "host_write requires contiguous tensor"

        return self._io_engine.submit_write(
            path, tensor, 0, tensor.numel() * tensor.element_size()
        )

    def host_read(self, file_id: str, tensor: Tensor, async_: bool = True) -> int:
        """Read from NVMe into a CPU tensor via io_uring.

        Returns:
            Handle for wait().
        """
        path = self._path(file_id)
        assert not tensor.is_cuda, "host_read requires a CPU tensor"
        assert tensor.is_contiguous(), "host_read requires contiguous tensor"

        return self._io_engine.submit_read(
            path, tensor, 0, tensor.numel() * tensor.element_size()
        )

    def wait(self, handle: int) -> None:
        """Wait for an async io_uring operation to complete."""
        self._io_engine.wait(handle)

    def wait_all(self) -> None:
        """Wait for all pending io_uring operations."""
        self._io_engine.wait_all()

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def exists(self, file_id: str) -> bool:
        return os.path.exists(self._path(file_id))

    def remove(self, file_id: str) -> None:
        path = self._path(file_id)
        if os.path.exists(path):
            os.remove(path)

    def cleanup(self) -> None:
        """Remove all files in storage directory."""
        if os.path.exists(self._storage_dir):
            shutil.rmtree(self._storage_dir)
            ensure_dir(self._storage_dir)

    def file_size(self, file_id: str) -> int:
        path = self._path(file_id)
        return os.path.getsize(path) if os.path.exists(path) else 0
