"""StorageTensor: a tensor container for host RAM and NVMe residency."""

from __future__ import annotations

from typing import Optional, Literal

import torch
from torch import Tensor

from grinnder.storage.backend import StorageBackend


class StorageTensor:
    """A storage-aware container for tensor data and residency metadata.

    The lifecycle:
      1. Created with a shape/dtype. Host tensor allocated on CPU.
      2. Data can be written (from GPU bypass, or from computation).
      3. ``to_storage()`` flushes host tensor to NVMe file.
      4. ``to_host()`` loads from NVMe file back to host tensor.
      5. ``to_gpu()`` copies host tensor to a provided GPU buffer.

    Memory management:
      - CPU memory is pageable by default. Pass ``pinned=True`` only for
        bounded transfer/staging buffers that benefit from pinned allocation.
      - When on storage: host tensor is ``resize_(0)`` to free RAM.
      - When on host: host tensor holds the data.
      - GPU copies are always transient (managed by DeviceBuffer).
    """

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        backend: StorageBackend,
        file_id: str,
        pinned: bool = False,
    ):
        self._shape = shape
        self._dtype = dtype
        self._backend = backend
        self._file_id = file_id
        self._element_size = torch.tensor([], dtype=dtype).element_size()
        self._numel = 1
        for s in shape:
            self._numel *= s
        self._nbytes = self._numel * self._element_size

        self._pinned = pinned and torch.cuda.is_available()
        self._host_tensor: Optional[Tensor] = self._make_host_tensor()
        self._location: Literal["host", "storage"] = "host"

        # Pending async I/O handle
        self._io_handle: int = -1

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def location(self) -> str:
        return self._location

    @property
    def file_id(self) -> str:
        return self._file_id

    @property
    def host_data(self) -> Tensor:
        """Access the host tensor. Must be on host."""
        assert self._location == "host" and self._host_tensor is not None, (
            f"Data is on {self._location}, call to_host() first"
        )
        return self._host_tensor

    def synchronize(self) -> None:
        """Wait for pending async I/O."""
        if self._io_handle >= 0:
            self._backend.wait(self._io_handle)
            self._io_handle = -1

    def to_storage(self, async_: bool = True) -> None:
        """Flush host tensor to NVMe via io_uring and free host memory."""
        if self._location == "storage":
            return
        self.synchronize()
        self._io_handle = self._backend.host_write(
            self._host_tensor, self._file_id
        )
        if not async_:
            self.synchronize()
            self._free_host()
            self._location = "storage"

    def finish_to_storage(self) -> None:
        """Complete async to_storage and free host memory."""
        self.synchronize()
        self._free_host()
        self._location = "storage"

    def to_host(self, async_: bool = True) -> None:
        """Load from NVMe to host memory via io_uring."""
        if self._location == "host":
            return
        self._alloc_host()
        self._io_handle = self._backend.host_read(
            self._file_id, self._host_tensor
        )
        if not async_:
            self.synchronize()
            self._location = "host"

    def finish_to_host(self) -> None:
        """Complete async to_host."""
        self.synchronize()
        self._location = "host"

    def to_gpu(
        self,
        target: Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Copy host tensor to a GPU buffer (non-blocking if stream provided)."""
        assert self._location == "host", "Must be on host to copy to GPU"
        if stream is not None:
            with torch.cuda.stream(stream):
                target.copy_(self._host_tensor, non_blocking=True)
        else:
            target.copy_(self._host_tensor)

    def fill_from_gpu(
        self,
        source: Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Copy GPU tensor to host buffer (D2H)."""
        assert self._location == "host", "Must be on host to receive D2H"
        if stream is not None:
            with torch.cuda.stream(stream):
                self._host_tensor.copy_(source, non_blocking=True)
        else:
            self._host_tensor.copy_(source)

    def bypass_from_gpu(
        self,
        source: Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Bypass: GPU tensor directly to storage (via GDS or temp host buffer)."""
        self._backend.gpu_write(source, self._file_id, stream)
        self._location = "storage"
        self._free_host()

    def _make_host_tensor(self) -> Tensor:
        """Create a new CPU tensor with the configured pinning policy."""
        return torch.empty(self._shape, dtype=self._dtype, pin_memory=self._pinned)

    def _alloc_host(self) -> None:
        """Re-allocate host tensor if freed."""
        if self._host_tensor is None:
            self._host_tensor = self._make_host_tensor()

    def _free_host(self) -> None:
        """Free host tensor memory."""
        self._host_tensor = None

    def reset(self) -> None:
        """Reset to initial state (host, zeroed)."""
        self._alloc_host()
        self._host_tensor.zero_()
        self._location = "host"
        self._io_handle = -1

    def __repr__(self) -> str:
        return (
            f"StorageTensor(shape={self._shape}, dtype={self._dtype}, "
            f"location={self._location}, file_id={self._file_id})"
        )
