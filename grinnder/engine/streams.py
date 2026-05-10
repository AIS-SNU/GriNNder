"""CUDA stream manager for double-buffered async transfers."""

from __future__ import annotations

from typing import List

import torch


class StreamManager:
    """Manages CUDA streams for overlapping I/O and computation.

    Four stream types:
      - compute: Main forward/backward computation stream.
      - h2d[pool_size]: Host-to-device feature/gradient uploads.
      - d2h[pool_size]: Device-to-host activation/gradient offloads.
      - act_h2d[pool_size]: Backward-pass activation regathering
        (separate from h2d to allow independent prefetching).

    Double buffering (pool_size=2) ensures that while partition i is
    being processed on GPU, partition i+1's data is being transferred.

    Args:
        pool_size: Number of stream slots for double buffering.
        device: CUDA device string.
    """

    def __init__(self, pool_size: int = 2, device: str = "cuda:0"):
        self.pool_size = pool_size
        self.device = device

        self.compute = torch.cuda.Stream(device)
        self.h2d: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device) for _ in range(pool_size)
        ]
        self.d2h: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device) for _ in range(pool_size)
        ]
        self.act_h2d: List[torch.cuda.Stream] = [
            torch.cuda.Stream(device) for _ in range(pool_size)
        ]
