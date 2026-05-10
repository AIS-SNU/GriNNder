"""GriNNder utility functions."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def fix_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_micro_f1(
    pred: Tensor, target: Tensor, mask: Optional[Tensor] = None
) -> float:
    """Compute micro-F1 (accuracy) for classification.

    Args:
        pred: Prediction logits [N, C] or [N].
        target: Ground truth labels [N].
        mask: Optional boolean mask [N].

    Returns:
        Accuracy as float.
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=-1)
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if target.numel() == 0:
        return 0.0
    return (pred == target).float().mean().item()


def get_available_host_memory() -> int:
    """Get available host memory in bytes.

    Uses /proc/meminfo on Linux, falls back to psutil.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (FileNotFoundError, ValueError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        return 0


def get_physical_cpu_count() -> int:
    """Return physical CPU cores when detectable, otherwise logical CPUs."""
    cpuinfo = "/proc/cpuinfo"
    try:
        cores = set()
        physical_id = None
        core_id = None
        with open(cpuinfo, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    if physical_id is not None and core_id is not None:
                        cores.add((physical_id, core_id))
                    physical_id = None
                    core_id = None
                    continue
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
        if physical_id is not None and core_id is not None:
            cores.add((physical_id, core_id))
        if cores:
            return len(cores)
    except OSError:
        pass
    return os.cpu_count() or 1


def get_default_partitioner_threads() -> int:
    """Thread default for CPU graph partitioning.

    Set ``GRINNDER_PARTITION_THREADS`` to override. Otherwise use all logical
    CPUs visible to the process.
    """
    env_value = os.environ.get("GRINNDER_PARTITION_THREADS")
    if env_value:
        return max(1, int(env_value))
    return os.cpu_count() or 1


def get_gpu_memory_used(device: int = 0) -> int:
    """Get GPU memory used in bytes."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated(device)


def get_gpu_memory_total(device: int = 0) -> int:
    """Get total GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(device)
    return props.total_memory


def report_memory(tag: str = "", device: int = 0) -> str:
    """Report GPU and host memory usage."""
    gpu_used = get_gpu_memory_used(device) / (1024**3)
    gpu_total = get_gpu_memory_total(device) / (1024**3)
    host_avail = get_available_host_memory() / (1024**3)
    msg = f"[{tag}] GPU: {gpu_used:.2f}/{gpu_total:.2f} GB | Host avail: {host_avail:.2f} GB"
    return msg


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path
