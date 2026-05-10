"""Host buffer: partition-wise tensor management on host RAM / NVMe storage.

Each HostBuffer manages one "layer" of activations or gradients, split into
per-partition tensors. Supports async gather, scatter, fill, upload, and
bypass-to-storage operations via C++ extensions.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grinnder.storage.backend import StorageBackend


def _load_ops():
    """Lazy-load C++ extension ops."""
    try:
        import grinnder._C as _C
        return _C
    except ImportError:
        return None


def _empty_host_tensor(
    rows: int,
    cols: int,
    *,
    pin_memory: bool,
    allocate: bool,
) -> Tensor:
    """Allocate a CPU tensor with optional pinned-memory allocation."""
    shape = (rows, cols) if allocate else (0, cols)
    try:
        return torch.empty(
            shape,
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
    except RuntimeError:
        if not pin_memory:
            raise
        return torch.empty(shape, dtype=torch.float32)


def _empty_pinned_host_cache() -> None:
    """Release cached pinned-host blocks when PyTorch exposes the hook."""
    empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if empty_cache is not None:
        empty_cache()


class HostBuffer:
    """Partition-wise tensor storage on host memory.

    Stores one tensor per partition. Supports async transfers to/from GPU
    and to/from NVMe storage. HostBuffer uses pageable CPU tensors by default;
    pass ``pin_memory=True`` to request pinned CPU tensors.

    Layout for gather:
        GPU tensor = [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]
        - intra: contiguous memcpy from self[pid]
        - boundary: index_select from self[src_pid] at boundary indices

    Args:
        num_parts: Number of partitions.
        part_sizes: Number of nodes in each partition.
        embedding_dim: Feature dimension.
        backend: StorageBackend for NVMe I/O (None = host-only).
        file_prefix: Prefix for NVMe file IDs.
    """

    def __init__(
        self,
        num_parts: int,
        part_sizes: List[int],
        embedding_dim: int,
        backend: Optional[StorageBackend] = None,
        file_prefix: str = "host",
        lazy: bool = True,
        pin_memory: bool = False,
    ):
        self.num_parts = num_parts
        self.part_sizes = part_sizes
        self.embedding_dim = embedding_dim
        self._backend = backend
        self._file_prefix = file_prefix
        self._element_size = torch.tensor([], dtype=torch.float32).element_size()
        self._lazy = lazy
        self._pin_memory = pin_memory
        self._zero_initialized = [False] * num_parts

        # Tensor objects keep their logical shape, while storage is allocated
        # only for partitions resident in host memory.
        self._tensors: List[Tensor] = []
        for i in range(num_parts):
            t = _empty_host_tensor(
                part_sizes[i],
                embedding_dim,
                pin_memory=pin_memory,
                allocate=not lazy,
            )
            if not lazy:
                t.zero_()
            self._tensors.append(t)

        self._ops = _load_ops()

    def __getitem__(self, pid: int) -> Tensor:
        """Get host tensor for partition pid, allocating it if needed."""
        self.allocate(pid)
        return self._tensors[pid]

    def __len__(self) -> int:
        return self.num_parts

    def partition_nbytes(self, pid: int) -> int:
        """Bytes required by one partition when resident in host RAM."""
        return self.part_sizes[pid] * self.embedding_dim * self._element_size

    @property
    def total_nbytes(self) -> int:
        return sum(self.partition_nbytes(pid) for pid in range(self.num_parts))

    def allocate(self, pid: int) -> Tensor:
        """Allocate host storage for partition pid if it is not resident."""
        t = self._tensors[pid]
        expected = self.partition_nbytes(pid)
        if t.untyped_storage().size() < expected:
            t.untyped_storage().resize_(expected)
        expected_shape = (self.part_sizes[pid], self.embedding_dim)
        if tuple(t.shape) != expected_shape:
            t.set_(t.untyped_storage(), 0, expected_shape, (self.embedding_dim, 1))
        return t

    def release(self, pid: int) -> None:
        """Release host storage for one partition while preserving tensor metadata."""
        self._tensors[pid].untyped_storage().resize_(0)
        self._zero_initialized[pid] = False
        if self._pin_memory:
            _empty_pinned_host_cache()

    def release_all(self) -> None:
        """Release host storage for every partition."""
        for pid in range(self.num_parts):
            self.release(pid)

    def is_allocated(self, pid: int) -> bool:
        """Return whether partition pid currently occupies host storage."""
        return self._tensors[pid].untyped_storage().size() >= self.partition_nbytes(pid)

    def allocated_pids(self) -> List[int]:
        """Return partitions that currently occupy host storage."""
        return [pid for pid in range(self.num_parts) if self.is_allocated(pid)]

    def zero_partition(self, pid: int) -> None:
        """Allocate and zero one partition buffer."""
        self.allocate(pid)
        self._tensors[pid].zero_()
        self._zero_initialized[pid] = True

    @property
    def resident_bytes(self) -> int:
        """Current host bytes occupied by resident partition storage."""
        return sum(
            self.partition_nbytes(pid)
            for pid in range(self.num_parts)
            if self.is_allocated(pid)
        )

    def storage_exists(self, pid: int) -> bool:
        """Return whether a backing storage copy exists for one partition."""
        if self._backend is None:
            return False
        return self._backend.exists(f"{self._file_prefix}_p{pid}")

    def ensure_storage_copy(self, pid: int) -> None:
        """Write a resident partition to storage if no backing copy exists."""
        if self._backend is None or self.storage_exists(pid):
            return
        if not self.is_allocated(pid):
            raise RuntimeError(
                f"HostBuffer partition {pid} is not resident for storage write"
            )
        self.cpu_to_storage(pid)

    # ------------------------------------------------------------------
    # GPU -> Host (D2H)
    # ------------------------------------------------------------------

    def async_fill(
        self, pid: int, gpu_data: Tensor, stream: torch.cuda.Stream
    ) -> None:
        """D2H: copy GPU tensor to host partition buffer.

        Args:
            pid: Partition index.
            gpu_data: GPU tensor [part_sizes[pid], dim].
            stream: CUDA stream for async copy.
        """
        assert gpu_data.is_cuda
        self.allocate(pid)
        stream.wait_stream(torch.cuda.current_stream(gpu_data.device))
        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.d2h_copy_async(gpu_data, self._tensors[pid])
            else:
                self._tensors[pid].copy_(gpu_data, non_blocking=True)
        self._zero_initialized[pid] = True

    def d2h_synchronize(self, stream: torch.cuda.Stream) -> None:
        """Wait for D2H operations to complete.

        Two-phase synchronization:
        1. C++ thread pool sync: wait for cudaMemcpyAsync calls to be submitted
        2. CUDA stream sync: wait for all submitted CUDA ops to complete
        """
        if self._ops is not None:
            try:
                self._ops.d2h_synchronize()
            except RuntimeError:
                pass  # No pending D2H operations
        torch.cuda.synchronize(stream)

    # ------------------------------------------------------------------
    # Host -> GPU (H2D)
    # ------------------------------------------------------------------

    def async_upload(
        self, pid: int, gpu_target: Tensor, stream: torch.cuda.Stream
    ) -> None:
        """H2D: copy host partition buffer to GPU tensor.

        Args:
            pid: Partition index.
            gpu_target: Pre-allocated GPU tensor.
            stream: CUDA stream for async copy.
        """
        assert gpu_target.is_cuda
        if not self.is_allocated(pid):
            raise RuntimeError(
                f"HostBuffer partition {pid} is not resident for upload"
            )
        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))
        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.h2d_copy_async(self._tensors[pid], gpu_target)
            else:
                gpu_target.copy_(self._tensors[pid], non_blocking=True)

    def h2d_synchronize(self, stream: torch.cuda.Stream) -> None:
        """Wait for H2D operations to complete.

        Two-phase synchronization:
        1. C++ thread pool sync: wait for cudaMemcpyAsync calls to be submitted
        2. CUDA stream sync: wait for all submitted CUDA ops to complete
        """
        if self._ops is not None:
            try:
                self._ops.h2d_synchronize()
            except RuntimeError:
                pass  # No pending H2D operations
        torch.cuda.synchronize(stream)

    # ------------------------------------------------------------------
    # Gather: multiple host partitions -> one GPU tensor
    # ------------------------------------------------------------------

    def async_gather(
        self,
        pid: int,
        gpu_target: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Gather features from host partitions to GPU.

        GPU layout: [intra(pid) | boundary_from_p0 | ... | boundary_from_pN]

        Args:
            pid: Target partition index.
            gpu_target: Pre-allocated GPU tensor [total_size, dim].
            boundaries: boundaries[src_pid] = index tensor into self[src_pid].
                        boundaries[pid] = None (intra-partition, copied contiguously).
            stream: CUDA stream.
        """
        assert gpu_target.is_cuda
        required_pids = {pid}
        for src_pid, boundary in enumerate(boundaries):
            if src_pid != pid and boundary is not None and boundary.numel() > 0:
                required_pids.add(src_pid)
        missing = [src_pid for src_pid in required_pids if not self.is_allocated(src_pid)]
        if missing:
            raise RuntimeError(
                f"HostBuffer partitions {missing} are not resident for gather"
            )

        # Build boundary list (replace None with empty tensor)
        bndries = []
        for i in range(self.num_parts):
            if i == pid or boundaries[i] is None:
                bndries.append(torch.empty(0, dtype=torch.long))
            else:
                bndries.append(boundaries[i])

        stream.wait_stream(torch.cuda.current_stream(gpu_target.device))
        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.gather_partitions(
                    pid, self._tensors, gpu_target, bndries
                )
            else:
                # Python fallback
                offset = self._tensors[pid].size(0)
                gpu_target[:offset].copy_(self._tensors[pid])
                for i in range(self.num_parts):
                    if i == pid or bndries[i].numel() == 0:
                        continue
                    selected = self._tensors[i].index_select(0, bndries[i])
                    n = selected.size(0)
                    gpu_target[offset : offset + n].copy_(selected)
                    offset += n

    # ------------------------------------------------------------------
    # Scatter: one GPU tensor -> multiple host partitions (with accumulation)
    # ------------------------------------------------------------------

    def async_scatter(
        self,
        pid: int,
        gpu_source: Tensor,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Scatter gradients from GPU to host partitions with accumulation.

        GPU layout matches gather: [intra(pid) | boundary_from_p0 | ...]
        Accumulates (+=) into existing host partition data.

        Args:
            pid: Source partition index.
            gpu_source: GPU tensor [total_size, dim].
            boundaries: Same format as gather.
            stream: CUDA stream.
        """
        assert gpu_source.is_cuda
        for src_pid, boundary in enumerate(boundaries):
            if src_pid == pid:
                self.allocate(src_pid)
                if not self._zero_initialized[src_pid]:
                    self._tensors[src_pid].zero_()
                    self._zero_initialized[src_pid] = True
            elif boundary is not None and boundary.numel() > 0:
                self.allocate(src_pid)
                if not self._zero_initialized[src_pid]:
                    self._tensors[src_pid].zero_()
                    self._zero_initialized[src_pid] = True

        bndries = []
        for i in range(self.num_parts):
            if i == pid or boundaries[i] is None:
                bndries.append(torch.empty(0, dtype=torch.long))
            else:
                bndries.append(boundaries[i])

        with torch.cuda.stream(stream):
            if self._ops is not None:
                self._ops.scatter_partitions(
                    pid, gpu_source, self._tensors, bndries
                )
            else:
                # Python fallback (synchronous)
                offset = self._tensors[pid].size(0)
                cpu_data = gpu_source[:offset].cpu()
                self._tensors[pid].add_(cpu_data)
                for i in range(self.num_parts):
                    if i == pid or bndries[i].numel() == 0:
                        continue
                    n = bndries[i].size(0)
                    cpu_slice = gpu_source[offset : offset + n].cpu()
                    self._tensors[i].index_put_(
                        (bndries[i],), cpu_slice, accumulate=True
                    )
                    offset += n

    # ------------------------------------------------------------------
    # Storage bypass: GPU -> NVMe directly
    # ------------------------------------------------------------------

    def async_bypass_to_storage(
        self, pid: int, gpu_data: Tensor, stream: torch.cuda.Stream
    ) -> None:
        """Bypass: write GPU tensor directly to NVMe storage.

        With GDS, kvikio uses a direct GPU-to-NVMe path. If direct GDS is not
        available, kvikio may use a compatible staging path internally.
        """
        assert self._backend is not None, "StorageBackend required for bypass"
        file_id = f"{self._file_prefix}_p{pid}"
        stream.wait_stream(torch.cuda.current_stream(gpu_data.device))
        stream.synchronize()
        self._backend.gpu_write(gpu_data, file_id, stream)

    # ------------------------------------------------------------------
    # Storage <-> Host (cache management)
    # ------------------------------------------------------------------

    def _ensure_allocated(self, pid: int) -> None:
        """Re-allocate host tensor if it was freed by cache eviction."""
        self.allocate(pid)

    def storage_to_cpu(self, pid: Optional[int] = None) -> None:
        """Load partition(s) from NVMe to host cache via io_uring.

        Re-allocates host tensor if previously freed by cache eviction.
        """
        assert self._backend is not None
        if pid is not None:
            self._ensure_allocated(pid)
            file_id = f"{self._file_prefix}_p{pid}"
            h = self._backend.host_read(file_id, self._tensors[pid])
            self._backend.wait(h)
            self._zero_initialized[pid] = True
        else:
            handles = []
            loaded = []
            for i in range(self.num_parts):
                file_id = f"{self._file_prefix}_p{i}"
                if self._backend.exists(file_id):
                    self._ensure_allocated(i)
                    h = self._backend.host_read(file_id, self._tensors[i])
                    handles.append(h)
                    loaded.append(i)
            for h in handles:
                self._backend.wait(h)
            for i in loaded:
                self._zero_initialized[i] = True

    def cpu_to_storage(self, pid: Optional[int] = None) -> None:
        """Flush partition(s) from host cache to NVMe via io_uring."""
        assert self._backend is not None
        if pid is not None:
            file_id = f"{self._file_prefix}_p{pid}"
            self.allocate(pid)
            h = self._backend.host_write(self._tensors[pid], file_id)
            self._backend.wait(h)
        else:
            handles = []
            for i in range(self.num_parts):
                file_id = f"{self._file_prefix}_p{i}"
                self.allocate(i)
                h = self._backend.host_write(self._tensors[i], file_id)
                handles.append(h)
            for h in handles:
                self._backend.wait(h)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def initialize_zeros(self, lazy: bool = False) -> None:
        """Zero all partition tensors (for gradient write-back init).

        Re-allocates any tensors that were freed by cache eviction.
        """
        if lazy:
            self.release_all()
            return
        for pid in range(self.num_parts):
            self._ensure_allocated(pid)
            self._tensors[pid].zero_()
            self._zero_initialized[pid] = True

    def reset(self) -> None:
        """Reset all tensors to zero."""
        self.initialize_zeros()
