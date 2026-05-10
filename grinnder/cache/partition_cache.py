"""Partition-granularity caching for host memory.

Manages which layers' activations are kept in host RAM vs NVMe storage.
Auto-selection uses the demand-managed modes:

  Mode 1 (lru_layer):     Whole-layer LRU, usually all-resident on large hosts
  Mode 2 (partition_lru): Per-partition LRU eviction (extreme scarcity)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

import torch

from grinnder.buffer.host import HostBuffer
from grinnder.storage.backend import StorageBackend
from grinnder.utils import get_available_host_memory


@dataclass(frozen=True)
class CacheMemoryPlan:
    """Concrete host-memory accounting used for cache-mode selection."""

    total_budget_bytes: int
    fixed_resident_bytes: int
    safety_margin_bytes: int
    remaining_cache_bytes: int
    activation_layer_bytes: Tuple[int, ...]
    gradient_layer_bytes: Tuple[int, ...]
    all_layer_residency_bytes: int
    layer_working_set_bytes: int
    largest_partition_dependency_bytes: int


class PartitionCache:
    """Partition-granularity caching with hierarchical replacement.

    Manages host memory for activations (forward gather, backward regather)
    and gradient write-back buffers. Auto-detects the best cache mode
    based on available host memory.

    Args:
        num_parts: Number of partitions.
        num_layers: Number of GNN layers.
        layer_dims: [in_dim, hid_dim, ..., out_dim] (len = num_layers + 1).
        num_nodes: Total number of nodes in graph.
        host_buffers: List of HostBuffers, one per layer (index 0..L).
        grad_buffers: List of HostBuffers for gradients (index 1..L-1, None for layer 0).
        backend: StorageBackend for NVMe I/O.
        mode: Cache mode or 'auto' for auto-detection.
        host_memory_budget_gb: Budget in GB (0 = auto-detect).
    """

    def __init__(
        self,
        num_parts: int,
        num_layers: int,
        layer_dims: List[int],
        num_nodes: int,
        host_buffers: List[HostBuffer],
        grad_buffers: List[Optional[HostBuffer]],
        backend: StorageBackend,
        mode: Literal["auto", "lru_layer", "partition_lru"] = "auto",
        host_memory_budget_gb: float = 0,
        fixed_resident_bytes: int = 0,
        safety_margin_bytes: int = 0,
        dependency_sets: Optional[List[Set[int]]] = None,
    ):
        valid_modes = {"auto", "lru_layer", "partition_lru"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported cache mode {mode!r}; expected one of "
                f"{sorted(valid_modes)}"
            )

        self.num_parts = num_parts
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.num_nodes = num_nodes
        self.host_buffers = host_buffers
        self.grad_buffers = grad_buffers
        self.backend = backend
        self.dependency_sets = dependency_sets

        if host_memory_budget_gb > 0:
            self._budget = int(host_memory_budget_gb * (1024**3))
        else:
            self._budget = get_available_host_memory()
        self._fixed_resident_bytes = max(0, int(fixed_resident_bytes))
        self._safety_margin_bytes = max(0, int(safety_margin_bytes))
        self._memory_plan = self._build_memory_plan()
        self._layer_activation_cache_budget = max(
            0,
            self._memory_plan.remaining_cache_bytes
            - max(self._memory_plan.gradient_layer_bytes or (0,)),
        )
        self._partition_activation_cache_budget = self._memory_plan.remaining_cache_bytes

        if mode == "auto":
            self._mode = self._auto_detect()
        else:
            self._mode = mode
        self._activation_cache_budget = (
            self._layer_activation_cache_budget
            if self._mode == "lru_layer"
            else self._partition_activation_cache_budget
        )

        # Track which layers are currently in host cache (layer-level tracking)
        self._cached_layers: OrderedDict[int, bool] = OrderedDict()

        # Track which (layer, partition) pairs are cached (partition-level tracking)
        # Used in partition_lru mode
        self._cached_partitions: OrderedDict[Tuple[int, int], bool] = OrderedDict()

        # Hit/miss counters
        self._hits = 0
        self._misses = 0
        self._resident_activation_bytes = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def memory_plan(self) -> CacheMemoryPlan:
        return self._memory_plan

    @property
    def remaining_cache_bytes(self) -> int:
        return self._memory_plan.remaining_cache_bytes

    @property
    def activation_cache_budget_bytes(self) -> int:
        """Host bytes available for resident activation cache entries."""
        return self._activation_cache_budget

    def _auto_detect(self) -> str:
        """Select cache mode based on available host memory.

        Host memory model:
          lru_layer:     cache holds full activation layers and evicts whole
                         layers only under budget pressure.
          partition_lru: cache holds dependency partitions and bypasses forward
                         outputs to storage.
          Always:        boundaries + per-partition adj temps.
        """
        available = self._layer_activation_cache_budget

        if available >= self._memory_plan.layer_working_set_bytes:
            return "lru_layer"
        return "partition_lru"

    def _build_memory_plan(self) -> CacheMemoryPlan:
        activation_layer_bytes = tuple(buf.total_nbytes for buf in self.host_buffers)
        gradient_layer_bytes = tuple(
            0 if buf is None else buf.total_nbytes for buf in self.grad_buffers
        )
        all_layer_residency_bytes = sum(activation_layer_bytes) + sum(
            gradient_layer_bytes
        )
        max_activation = max(activation_layer_bytes or (0,))
        max_gradient = max(gradient_layer_bytes or (0,))
        layer_working_set_bytes = max_activation + max_gradient
        largest_dependency = self._largest_partition_dependency_bytes()
        remaining = max(
            0,
            self._budget - self._fixed_resident_bytes - self._safety_margin_bytes,
        )
        return CacheMemoryPlan(
            total_budget_bytes=self._budget,
            fixed_resident_bytes=self._fixed_resident_bytes,
            safety_margin_bytes=self._safety_margin_bytes,
            remaining_cache_bytes=remaining,
            activation_layer_bytes=activation_layer_bytes,
            gradient_layer_bytes=gradient_layer_bytes,
            all_layer_residency_bytes=all_layer_residency_bytes,
            layer_working_set_bytes=layer_working_set_bytes,
            largest_partition_dependency_bytes=largest_dependency,
        )

    def _largest_partition_dependency_bytes(self) -> int:
        largest = 0
        layer_id = self._largest_activation_layer_id()
        for pid in range(self.num_parts):
            deps = self._dependency_set(pid)
            total = sum(self._partition_bytes(layer_id, src_pid) for src_pid in deps)
            largest = max(largest, total)
        return largest

    def _largest_activation_layer_id(self) -> int:
        sizes = [buf.total_nbytes for buf in self.host_buffers]
        return max(range(len(sizes)), key=lambda i: sizes[i]) if sizes else 0

    def ensure_in_host(self, layer_id: int) -> None:
        """Ensure layer's activations are in host cache for gather/regather.

        lru_layer: load full layer from storage, evict LRU layer if needed.
        partition_lru: individual partitions loaded on demand via
               ensure_partition_in_host(). This method pre-loads all partitions
               of the layer, evicting other (layer, partition) pairs as needed.
        """
        if self._mode == "lru_layer":
            if layer_id in self._cached_layers:
                self._cached_layers.move_to_end(layer_id)
                self._hits += 1
                return

            self._misses += 1

            # Evict oldest layer(s) until budget allows new layer
            layer_bytes = self._layer_bytes(layer_id)
            if layer_bytes > self._activation_cache_budget:
                raise MemoryError(
                    f"Layer {layer_id} requires {layer_bytes} bytes, but only "
                    f"{self._activation_cache_budget} activation-cache bytes remain"
                )
            while (
                self._resident_activation_bytes + layer_bytes
                > self._activation_cache_budget
            ):
                if not self._cached_layers:
                    raise MemoryError(
                        f"Cannot make room for layer {layer_id} within "
                        f"{self._activation_cache_budget} bytes"
                    )
                evict_id, _ = self._cached_layers.popitem(last=False)
                self._evict_layer(evict_id)

            # Load full layer from storage only for partitions that are not
            # already resident. Newly produced layers may still be in host RAM.
            for pid in range(self.num_parts):
                if not self.host_buffers[layer_id].is_allocated(pid):
                    self.host_buffers[layer_id].storage_to_cpu(pid=pid)
            self._cached_layers[layer_id] = True
            self._resident_activation_bytes += layer_bytes
            return

        if self._mode == "partition_lru":
            # Compatibility path: callers that do not provide a target partition
            # request the whole layer. The trainer uses dependency-aware calls.
            for pid in range(self.num_parts):
                self.ensure_partition_in_host(layer_id, pid)
            return

    def ensure_partition_in_host(self, layer_id: int, pid: int) -> None:
        """Ensure a single (layer, partition) is in host cache.

        Used in partition_lru mode. Evicts least-recently-used (layer, partition)
        pairs when host memory budget is exceeded.
        """
        self.ensure_dependencies_in_host(layer_id, pid, dependencies={pid})

    def ensure_dependencies_in_host(
        self,
        layer_id: int,
        target_pid: int,
        boundaries: Optional[List[Optional[torch.Tensor]]] = None,
        dependencies: Optional[Set[int]] = None,
    ) -> None:
        """Ensure the dependency partitions for a target partition are resident.

        Partition-wise replacement is demand-aware: all partitions needed for the
        current target partition's gather/regather are protected from eviction,
        while older unrelated partitions are evicted until the dependency set
        fits the remaining activation-cache budget.
        """
        deps = self._dependency_set(target_pid, boundaries, dependencies)
        keys = {(layer_id, pid) for pid in deps}

        demand_bytes = sum(self._partition_bytes(layer_id, pid) for pid in deps)
        if demand_bytes > self._activation_cache_budget:
            raise MemoryError(
                f"Layer {layer_id}, partition {target_pid} dependencies require "
                f"{demand_bytes} bytes, but only {self._activation_cache_budget} "
                "activation-cache bytes remain"
            )

        missing = []
        for key in sorted(keys):
            if key in self._cached_partitions:
                self._cached_partitions.move_to_end(key)
                self._hits += 1
            else:
                missing.append(key)
                self._misses += 1

        missing_bytes = sum(self._partition_bytes(layer_id, pid) for _, pid in missing)
        while (
            self._resident_activation_bytes + missing_bytes
            > self._activation_cache_budget
        ):
            evict_key = self._pop_evictable_partition(keys)
            if evict_key is None:
                raise MemoryError(
                    f"Cannot make room for layer {layer_id}, partition "
                    f"{target_pid} dependency set within "
                    f"{self._activation_cache_budget} bytes"
                )
            self._evict_partition(evict_key[0], evict_key[1])

        for key in missing:
            _, pid = key
            self.host_buffers[layer_id].storage_to_cpu(pid=pid)
            self._cached_partitions[key] = True
            self._resident_activation_bytes += self._partition_bytes(layer_id, pid)

    def on_layer_complete(self, layer_id: int) -> None:
        """Called after forward layer completes. Decide what to flush."""
        output_layer = layer_id + 1
        if output_layer >= len(self.host_buffers):
            return

        if self._mode == "lru_layer":
            if output_layer in self._cached_layers:
                self._cached_layers.move_to_end(output_layer)
                return

            layer_bytes = self._layer_bytes(output_layer)
            if layer_bytes > self._activation_cache_budget:
                raise MemoryError(
                    f"Layer {output_layer} requires {layer_bytes} bytes, but "
                    f"only {self._activation_cache_budget} activation-cache "
                    "bytes remain"
                )
            while (
                self._resident_activation_bytes + layer_bytes
                > self._activation_cache_budget
            ):
                if not self._cached_layers:
                    raise MemoryError(
                        f"Cannot keep layer {output_layer} within "
                        f"{self._activation_cache_budget} bytes"
                    )
                evict_id, _ = self._cached_layers.popitem(last=False)
                self._evict_layer(evict_id)

            self._cached_layers[output_layer] = True
            self._resident_activation_bytes += layer_bytes
            return

        # Partition-wise mode still bypasses outputs to storage and loads only
        # demanded partitions on the next gather/regather.

    def on_backward_layer_complete(self, layer_id: int) -> None:
        """Called after backward layer completes. Flush gradients to storage."""
        if self.grad_buffers[layer_id] is not None:
            if self._mode == "partition_lru":
                for pid in self.grad_buffers[layer_id].allocated_pids():
                    self.grad_buffers[layer_id].cpu_to_storage(pid)
                    self.grad_buffers[layer_id].release(pid)
                return
            self.grad_buffers[layer_id].cpu_to_storage()

    def prepare_layer(self, layer_id: int) -> None:
        """Alias for ensure_in_host (used in backward for regathering)."""
        self.ensure_in_host(layer_id)

    def _evict_layer(self, layer_id: int) -> None:
        """Evict a layer's activations from host cache.

        Ensure a storage copy exists before freeing host memory. For true
        layer-wise LRU, newly produced layers stay resident until budget
        pressure actually evicts them.
        """
        buf = self.host_buffers[layer_id]
        for pid in range(self.num_parts):
            buf.ensure_storage_copy(pid)
            buf.release(pid)
        self._resident_activation_bytes = max(
            0, self._resident_activation_bytes - self._layer_bytes(layer_id)
        )

    def _evict_partition(self, layer_id: int, pid: int) -> None:
        """Evict a single partition's activations from host cache.

        Ensure a storage copy exists before freeing host memory.
        """
        key = (layer_id, pid)
        buf = self.host_buffers[layer_id]
        buf.ensure_storage_copy(pid)
        buf.release(pid)
        self._resident_activation_bytes = max(
            0, self._resident_activation_bytes - self._partition_bytes(layer_id, pid)
        )
        self._cached_partitions.pop(key, None)

    def _layer_bytes(self, layer_id: int) -> int:
        """Estimate bytes for one layer's activations."""
        return self.host_buffers[layer_id].total_nbytes

    def _partition_bytes(self, layer_id: int, pid: int) -> int:
        return self.host_buffers[layer_id].partition_nbytes(pid)

    def _dependency_set(
        self,
        target_pid: int,
        boundaries: Optional[List[Optional[torch.Tensor]]] = None,
        dependencies: Optional[Set[int]] = None,
    ) -> Set[int]:
        if dependencies is not None:
            return set(dependencies)
        if boundaries is not None:
            deps = {target_pid}
            for src_pid, boundary in enumerate(boundaries):
                if (
                    src_pid != target_pid
                    and boundary is not None
                    and boundary.numel() > 0
                ):
                    deps.add(src_pid)
            return deps
        if self.dependency_sets is not None:
            return set(self.dependency_sets[target_pid])
        return {target_pid}

    def _pop_evictable_partition(
        self, protected: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        for key in list(self._cached_partitions.keys()):
            if key not in protected:
                self._cached_partitions.pop(key)
                return key
        return None

    def reset(self) -> None:
        """Reset cache state for new epoch."""
        if self._mode == "lru_layer":
            for layer_id in list(self._cached_layers.keys()):
                self._evict_layer(layer_id)
        elif self._mode == "partition_lru":
            for layer_id, pid in list(self._cached_partitions.keys()):
                self._evict_partition(layer_id, pid)
        self._cached_layers.clear()
        self._cached_partitions.clear()
        self._hits = 0
        self._misses = 0
        self._resident_activation_bytes = 0
