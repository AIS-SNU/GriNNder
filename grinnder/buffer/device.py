"""Device buffer: partition-wise GPU tensor management with double buffering.

Only pool_size (default 2) partition slots are allocated at any time.
After processing, partitions are freed via resize_(0).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from grinnder.buffer.host import HostBuffer


class DeviceBuffer:
    """Partition-wise tensor storage on GPU.

    Manages per-partition GPU tensors with lazy allocation:
    - allocate(pid): resize storage to hold partition data
    - release(pid): resize storage to 0 (free GPU memory)

    Only pool_size partitions should be live at any time for double buffering.

    Args:
        num_parts: Number of partitions.
        part_sizes: Number of nodes (rows) per partition. For gathered tensors,
                    this should be the EXPANDED size (intra + boundary nodes).
        embedding_dim: Feature dimension.
        device: CUDA device string.
    """

    def __init__(
        self,
        num_parts: int,
        part_sizes: List[int],
        embedding_dim: int,
        device: str = "cuda:0",
    ):
        self.num_parts = num_parts
        self.part_sizes = part_sizes
        self.embedding_dim = embedding_dim
        self.device = device

        # Pre-allocate tensor objects with zero storage (lazy allocation)
        self._tensors: List[Tensor] = []
        for i in range(num_parts):
            t = torch.empty(
                part_sizes[i], embedding_dim, dtype=torch.float32, device=device
            )
            t.untyped_storage().resize_(0)  # Free memory, keep tensor object
            self._tensors.append(t)

    def __getitem__(self, pid: int) -> Tensor:
        """Get GPU tensor for partition pid. Must be allocated first."""
        return self._tensors[pid]

    def __len__(self) -> int:
        return self.num_parts

    def allocate(self, pid: int) -> Tensor:
        """Allocate GPU memory for partition pid.

        Returns the allocated tensor.
        """
        t = self._tensors[pid]
        nbytes = t.numel() * t.element_size()
        if t.untyped_storage().size() < nbytes:
            t.untyped_storage().resize_(nbytes)
        return t

    def release(self, pid: int) -> None:
        """Free GPU memory for partition pid (resize storage to 0)."""
        tensor = self._tensors[pid]
        tensor.untyped_storage().resize_(0)
        tensor.grad = None

    def is_allocated(self, pid: int) -> bool:
        """Check if partition pid has GPU memory allocated."""
        return self._tensors[pid].untyped_storage().size() > 0

    def allocated_pids(self) -> List[int]:
        """Return partitions that currently occupy GPU storage."""
        return [pid for pid in range(self.num_parts) if self.is_allocated(pid)]

    @property
    def resident_bytes(self) -> int:
        """Current GPU bytes occupied by resident partition storage."""
        return sum(
            t.untyped_storage().size()
            for t in self._tensors
            if t.untyped_storage().size() > 0
        )

    # ------------------------------------------------------------------
    # Gather shortcut: allocate + gather from HostBuffer
    # ------------------------------------------------------------------

    def async_gather(
        self,
        pid: int,
        host_buffer: HostBuffer,
        boundaries: List[Optional[Tensor]],
        stream: torch.cuda.Stream,
    ) -> None:
        """Allocate GPU tensor and gather from host partitions.

        Combines allocate() + host_buffer.async_gather() for convenience.
        """
        self.allocate(pid)
        host_buffer.async_gather(pid, self._tensors[pid], boundaries, stream)

    def h2d_synchronize(self, stream: torch.cuda.Stream) -> None:
        """Wait for H2D operations to complete.

        Two-phase sync: C++ thread pool + CUDA stream.
        """
        try:
            from grinnder._C import h2d_synchronize as _h2d_sync
            try:
                _h2d_sync()
            except RuntimeError:
                pass  # No pending H2D operations
        except ImportError:
            pass
        torch.cuda.synchronize(stream)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Free all GPU memory."""
        for i in range(self.num_parts):
            self.release(i)
