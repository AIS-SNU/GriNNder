"""GriNNder configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Literal


def _default_threads() -> int:
    env_value = os.environ.get("GRINNDER_PARTITION_THREADS")
    if env_value:
        return max(1, int(env_value))
    return os.cpu_count() or 1


@dataclass
class GriNNderConfig:
    """Configuration for GriNNder training.

    Training Modes:
        'hongtu':   Baseline. Standard PyTorch activation checkpointing with
                    alpha-fold redundant snapshots in host RAM. When host memory
                    overflows, OS swap kicks in (unmanaged, random I/O).
        'grinnder': Full Structured Storage Offloading (SSO). Regathering
                    eliminates snapshot redundancy. Application-managed NVMe
                    storage with partition-wise caching. Cache auto-adapts to
                    available host memory:
                      - layer working set fits -> lru_layer
                      - tight RAM -> partition_lru

    Cache Modes (only used when mode='grinnder'):
        'auto':          Select lru_layer, then partition_lru under pressure.
        'lru_layer':     Evict whole layers in LRU order.
        'partition_lru': Evict individual partitions in LRU order.
    """

    # Training mode
    mode: Literal["hongtu", "grinnder"] = "grinnder"

    # Partitioning (delegates to grdpart)
    num_parts: int = 8
    partitioner: Literal["grinnder", "spinner"] = "grinnder"
    partitioner_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "capacity": 1.1,
            "beta": 1.0,
            "max_iter": 50,
            "num_threads": _default_threads(),
        }
    )

    # Storage (used when mode='grinnder')
    storage_dir: str = "/pci5_nvme/grinnder"

    # Cache replacement policy (used when mode='grinnder')
    cache_mode: Literal["auto", "lru_layer", "partition_lru"] = "auto"
    host_memory_budget_gb: float = 0  # 0 = auto-detect
    runtime_safety_margin_gb: float = 0.0

    # System
    device: str = "cuda:0"
    pool_size: int = 2  # Double buffering
    preprocess_workers: int = field(default_factory=_default_threads)

    # Multi-GPU
    num_gpus: int = 1
