"""Activation checkpointing via saved_tensors_hooks.

Both classes inherit from saved_tensors_hooks and are used with
torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=True).

IMPORTANT: adj must be loaded INSIDE the checkpointed function (not passed
to checkpoint), so checkpoint only saves x (gathered features). The hooks
intercept only x's save/restore.

HongtuCheckpoint ('cpu' strategy):
  pack: copy x to CPU
  unpack: copy back into device_buffer[pid]

ScatteredCheckpoint ('scattered' strategy):
  pack: save device_buffer reference only (no data copy, no redundant snapshot)
  unpack: wait for trainer's backward H2D re-gather, return device_buffer[pid]
  Eliminates alpha-fold data amplification.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.autograd.graph import saved_tensors_hooks

from grinnder.buffer.device import DeviceBuffer


class HongtuCheckpoint(saved_tensors_hooks):
    """HongTu baseline: saves gathered features to CPU during forward,
    copies back to device_buffer during backward recompute.

    Args:
        pid: Partition index.
        device_buffer: DeviceBuffer holding gathered activations.
    """

    def __init__(self, pid: int, device_buffer: DeviceBuffer):
        def pack(tensor: Tensor):
            return (tensor.device, tensor.cpu(), device_buffer)

        def unpack(packed):
            device, cpu_tensor, dbuf = packed
            dbuf.allocate(pid)
            dbuf[pid].copy_(cpu_tensor)
            return dbuf[pid]

        super().__init__(pack, unpack)


class ScatteredCheckpoint(saved_tensors_hooks):
    """GriNNder's scattered checkpointing: no snapshot redundancy.

    pack: assert tensor is device_buffer[pid], save reference only.
    unpack: trainer's backward prefetch has re-gathered compact activations
            from host into device_buffer[pid]. Wait for H2D stream, return it.

    Args:
        pid: Partition index.
        device_buffer: DeviceBuffer holding gathered activations.
        h2d_stream: CUDA stream for backward activation re-gather.
    """

    def __init__(
        self,
        pid: int,
        device_buffer: DeviceBuffer,
        h2d_stream: torch.cuda.Stream,
    ):
        def pack(tensor: Tensor):
            assert tensor is device_buffer[pid], (
                "ScatteredCheckpoint: saved tensor is not device_buffer[pid]"
            )
            return (tensor.device, device_buffer, h2d_stream)

        def unpack(packed):
            device, dbuf, stream = packed
            dbuf.h2d_synchronize(stream)
            return dbuf[pid]

        super().__init__(pack, unpack)
