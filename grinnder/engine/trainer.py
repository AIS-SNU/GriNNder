"""Trainer: central orchestrator for GriNNder's SSO training pipeline.

Owns all buffers, streams, and cache. Implements the full layer-wise
forward/backward with double buffering, as described in Algorithm 1
of the MLSys 2026 paper.
"""

from __future__ import annotations

import time
import gc
from typing import Dict, List, Optional, Any, Set

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from grinnder.config import GriNNderConfig
from grinnder.data.graph import PartitionedGraph
from grinnder.nn.base import GriNNderModel
from grinnder.buffer.host import HostBuffer
from grinnder.buffer.device import DeviceBuffer
from grinnder.autograd.grad_offload import GradOffload
from grinnder.autograd.checkpoint import ScatteredCheckpoint, HongtuCheckpoint

from grinnder.cache.partition_cache import PartitionCache
from grinnder.engine.streams import StreamManager
from grinnder.storage.backend import StorageBackend
from grinnder.utils import compute_micro_f1


class Trainer:
    """Orchestrates full-graph GNN training with Structured Storage Offloading.

    Implements the full training pipeline:
      1. Layer-wise forward with partition-wise gather + bypass
      2. Double-buffered loss computation
      3. Layer-wise backward with regather + scatter-accumulate
      4. Weight update

    Args:
        model: GriNNderModel instance.
        graph: PartitionedGraph from build_partitioned_graph().
        config: GriNNderConfig.
    """

    def __init__(
        self,
        model: GriNNderModel,
        graph: PartitionedGraph,
        config: GriNNderConfig,
        partition_range: Optional[tuple] = None,
    ):
        self.model = model
        self.graph = graph
        self.config = config
        self.device = config.device
        self._log_progress = graph.num_nodes >= 1_000_000

        # Partition range for multi-GPU: (start_pid, end_pid)
        # None = process all partitions (single GPU)
        if partition_range is not None:
            self._pid_start, self._pid_end = partition_range
        else:
            self._pid_start, self._pid_end = 0, graph.num_parts
        self._my_pids = list(range(self._pid_start, self._pid_end))

        dims = model.layer_dims()
        assert len(dims) == model.num_layers + 1
        num_layers = model.num_layers
        num_parts = graph.num_parts

        # Storage backend
        self.backend = StorageBackend(config.storage_dir) if config.mode == "grinnder" else None

        # CUDA streams
        self.streams = StreamManager(pool_size=config.pool_size, device=config.device)

        # HostBuffers for activations: one per layer (0..L)
        # Layer 0 = input features, Layers 1..L = output of each layer
        self.host_features: List[HostBuffer] = []
        for l in range(num_layers + 1):
            self.host_features.append(
                HostBuffer(
                    num_parts, graph.partition_sizes, dims[l],
                    backend=self.backend,
                    file_prefix=f"feat_l{l}",
                )
            )

        # HostBuffers for gradient write-back: one per layer (1..L-1)
        # Layer 0 has no gradient (input features). Layer L gradients flow through loss.
        self.host_gradients: List[Optional[HostBuffer]] = [None]  # layer 0
        for l in range(1, num_layers):
            self.host_gradients.append(
                HostBuffer(
                    num_parts, graph.partition_sizes, dims[l],
                    backend=self.backend,
                    file_prefix=f"grad_l{l}",
                )
            )

        # DeviceBuffers for gathered features (expanded sizes including boundaries)
        self.device_features: List[DeviceBuffer] = []
        for l in range(num_layers):
            self.device_features.append(
                DeviceBuffer(
                    num_parts, graph.expanded_sizes, dims[l],
                    device=config.device,
                )
            )

        # DeviceBuffers for gradient upload
        self.device_gradients: List[Optional[DeviceBuffer]] = [None]  # layer 0
        for l in range(1, num_layers):
            self.device_gradients.append(
                DeviceBuffer(
                    num_parts, graph.partition_sizes, dims[l],
                    device=config.device,
                )
            )

        # Activation references (for loss computation + backward)
        self.activations: List[List[Optional[Tensor]]] = [
            [None] * num_parts for _ in range(num_layers)
        ]

        # Mapping from partition ID to loss index (set by _compute_losses)
        self._pid_to_loss_idx: Dict[int, int] = {}

        # Partition cache (grinnder mode only)
        self.cache: Optional[PartitionCache] = None
        if config.mode == "grinnder":
            fixed_resident_bytes = self._estimate_fixed_resident_bytes(graph)
            safety_margin_bytes = int(config.runtime_safety_margin_gb * (1024**3))
            self.cache = PartitionCache(
                num_parts=num_parts,
                num_layers=num_layers,
                layer_dims=dims,
                num_nodes=graph.num_nodes,
                host_buffers=self.host_features,
                grad_buffers=self.host_gradients,
                backend=self.backend,
                mode=config.cache_mode,
                host_memory_budget_gb=config.host_memory_budget_gb,
                fixed_resident_bytes=fixed_resident_bytes,
                safety_margin_bytes=safety_margin_bytes,
                dependency_sets=self._dependency_sets(graph),
            )

        self._single_partition_fast_path_enabled = (
            graph.num_parts == 1
            and partition_range is None
            and (
                config.mode == "hongtu"
                or (
                    config.mode == "grinnder"
                    and self.cache is not None
                    and self.cache.mode == "lru_layer"
                )
            )
        )

        # Pre-fill host_features[0] with initial node features
        if not self._single_partition_fast_path_enabled:
            self._prefill_features()

    def _progress(self, message: str) -> None:
        if self._log_progress:
            print(f"      {message}{self._progress_memory_suffix()}", flush=True)

    def _progress_memory_suffix(self) -> str:
        if not torch.cuda.is_available():
            return ""
        alloc = torch.cuda.memory_allocated(self.device) / float(1024**3)
        reserved = torch.cuda.memory_reserved(self.device) / float(1024**3)
        device_resident = self._device_buffer_resident_bytes() / float(1024**3)
        return (
            f" gpu_alloc_gb={alloc:.2f}"
            f" gpu_reserved_gb={reserved:.2f}"
            f" device_buffers_gb={device_resident:.2f}"
        )

    def _device_buffer_resident_bytes(self) -> int:
        total = sum(buf.resident_bytes for buf in self.device_features)
        total += sum(
            0 if buf is None else buf.resident_bytes for buf in self.device_gradients
        )
        return total

    def _prefill_features(self) -> None:
        """Load initial node features into host_features[0]."""
        log_progress = self.graph.num_nodes >= 1_000_000
        if log_progress:
            print(
                f"    prefill input features start "
                f"parts={self.graph.num_parts} feat_dim={self.graph.feat_dim}",
                flush=True,
            )
        for pid in range(self.graph.num_parts):
            start = time.perf_counter()
            feat = self.graph.partition_features(pid)
            self.host_features[0][pid].copy_(feat)
            del feat
            gc.collect()
            if self._uses_partition_lru():
                self.host_features[0].cpu_to_storage(pid)
                self.host_features[0].release(pid)
            if log_progress:
                print(
                    f"    prefill input features {pid + 1}/{self.graph.num_parts} "
                    f"time_s={time.perf_counter() - start:.3f}",
                    flush=True,
                )
        if log_progress:
            print("    prefill input features done", flush=True)

    @staticmethod
    def _tensor_nbytes(tensor: Optional[Any]) -> int:
        if tensor is None:
            return 0
        if hasattr(tensor, "resident_nbytes"):
            return int(tensor.resident_nbytes)
        if not isinstance(tensor, torch.Tensor):
            return 0
        return tensor.numel() * tensor.element_size()

    @classmethod
    def _estimate_fixed_resident_bytes(cls, graph: PartitionedGraph) -> int:
        """Estimate graph metadata that stays resident outside the cache."""
        total = 0
        total += cls._tensor_nbytes(graph.features)
        total += cls._tensor_nbytes(graph.labels)
        total += cls._tensor_nbytes(graph.train_mask)
        total += cls._tensor_nbytes(graph.val_mask)
        total += cls._tensor_nbytes(graph.test_mask)
        total += cls._tensor_nbytes(graph.perm)
        total += cls._tensor_nbytes(graph.ptr)
        total += len(graph.partition_sizes) * 8

        for boundaries in graph.boundaries:
            for boundary in boundaries:
                total += cls._tensor_nbytes(boundary)

        for rowptr, col, value in graph.adj_csr:
            total += cls._tensor_nbytes(rowptr)
            total += cls._tensor_nbytes(col)
            total += cls._tensor_nbytes(value)

        return total

    @staticmethod
    def _dependency_sets(graph: PartitionedGraph) -> List[Set[int]]:
        deps: List[Set[int]] = []
        for pid, boundaries in enumerate(graph.boundaries):
            dep = {pid}
            for src_pid, boundary in enumerate(boundaries):
                if src_pid != pid and boundary is not None and boundary.numel() > 0:
                    dep.add(src_pid)
            deps.append(dep)
        return deps

    def _uses_partition_lru(self) -> bool:
        return self.cache is not None and self.cache.mode == "partition_lru"

    def _ensure_gradient_partition_resident(self, layer_id: int, pid: int) -> None:
        """Load a gradient partition from storage, or materialize zero if absent."""
        grad = self.host_gradients[layer_id]
        if grad is None or grad.is_allocated(pid):
            return
        if grad.storage_exists(pid):
            grad.storage_to_cpu(pid)
        else:
            grad.zero_partition(pid)

    def _prepare_cache_layer(self, layer_id: int) -> None:
        if self.cache is not None and self.cache.mode != "partition_lru":
            self.cache.ensure_in_host(layer_id)

    def _prepare_cache_partition(self, layer_id: int, pid: int) -> None:
        if self._uses_partition_lru():
            self.cache.ensure_dependencies_in_host(
                layer_id,
                pid,
                boundaries=self.graph.boundaries[pid],
            )

    def _get_adj(self, pid: int):
        """Get SparseTensor adjacency for partition pid."""
        from torch_sparse import SparseTensor

        rowptr, col, value = self.graph.adj_csr[pid]
        return SparseTensor(
            rowptr=rowptr.to(self.device),
            col=col.to(self.device),
            value=value.to(self.device) if value is not None else None,
            sparse_sizes=(
                self.graph.partition_sizes[pid],
                self.graph.expanded_sizes[pid],
            ),
            is_sorted=True,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def reset_epoch(self) -> None:
        """Reset all buffers for a new epoch."""
        partition_lru = self._uses_partition_lru()
        if partition_lru and self.cache is not None:
            self.cache.reset()

        for l in range(1, len(self.host_features)):
            if partition_lru:
                self.host_features[l].release_all()
            elif self.cache is None:
                self.host_features[l].reset()
        for l in range(1, len(self.host_gradients)):
            if self.host_gradients[l] is not None:
                if partition_lru:
                    self.host_gradients[l].release_all()
                elif self.cache is None:
                    self.host_gradients[l].reset()
        for db in self.device_features:
            db.reset()
        for db in self.device_gradients:
            if db is not None:
                db.reset()
        for layer_acts in self.activations:
            for i in range(len(layer_acts)):
                layer_acts[i] = None

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> Dict[str, float]:
        """Run one full training epoch.

        Returns dict with 'loss', 'train_acc', 'val_acc', 'test_acc'.
        """
        self.model.train()
        if self._use_single_partition_fast_path():
            return self._train_single_partition_fast_path(optimizer)

        self.reset_epoch()
        optimizer.zero_grad()

        # Phase 1: Layer-wise forward
        for layer_id in range(self.model.num_layers):
            self._progress(
                f"forward layer {layer_id + 1}/{self.model.num_layers} start"
            )
            self._forward_layer(layer_id)
            self._progress(
                f"forward layer {layer_id + 1}/{self.model.num_layers} done"
            )

        # Phase 2: compute per-partition losses. In bypass modes the final
        # activations are loaded from storage one partition at a time.
        self._progress("loss start")
        losses, metrics = self._compute_losses(criterion)
        self._progress("loss done")

        # Phase 3: Layer-wise backward (reverse)
        # Losses are sum-reduced per partition. Backward accumulates gradients.
        self._progress(f"backward layer {self.model.num_layers}/{self.model.num_layers} start")
        self._backward_last_layer(losses)
        self._progress(f"backward layer {self.model.num_layers}/{self.model.num_layers} done")
        for layer_id in reversed(range(self.model.num_layers - 1)):
            self._progress(
                f"backward layer {layer_id + 1}/{self.model.num_layers} start"
            )
            self._backward_layer(layer_id)
            self._progress(
                f"backward layer {layer_id + 1}/{self.model.num_layers} done"
            )

        # Scale gradients by 1/total_train_nodes for mean reduction
        # (equivalent to CrossEntropyLoss(reduction='mean') over all nodes)
        n_total_train = metrics.get("_n_train", 1)
        if n_total_train > 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(n_total_train)

        # Phase 4: Weight update
        self._progress("optimizer step start")
        optimizer.step()
        self._progress("optimizer step done")

        return metrics

    @torch.no_grad()
    def evaluate(self, criterion: torch.nn.Module) -> Dict[str, float]:
        """Evaluate model (forward only, no backward)."""
        self.model.eval()
        if self._use_single_partition_fast_path():
            return self._evaluate_single_partition_fast_path()

        self.reset_epoch()

        for layer_id in range(self.model.num_layers):
            self._progress(
                f"eval forward layer {layer_id + 1}/{self.model.num_layers} start"
            )
            self._forward_layer(layer_id)
            self._progress(
                f"eval forward layer {layer_id + 1}/{self.model.num_layers} done"
            )

        self._progress("eval loss start")
        _, metrics = self._compute_losses(criterion)
        self._progress("eval loss done")
        return metrics

    def _use_single_partition_fast_path(self) -> bool:
        return self._single_partition_fast_path_enabled

    def _single_partition_logits(self) -> Tensor:
        x = self.graph.partition_features(0).to(self.device)
        adj = self._get_adj(0)
        out = x
        for layer_id in range(self.model.num_layers):
            out = self.model.forward_layer(layer_id, out, adj)
        del adj
        return out

    def _single_partition_metrics(
        self, out: Tensor, y: Tensor
    ) -> tuple[Tensor, Dict[str, float]]:
        train_mask = self.graph.partition_train_mask(0).to(self.device)
        val_mask = self.graph.partition_val_mask(0).to(self.device)
        test_mask = self.graph.partition_test_mask(0).to(self.device)
        n_train = int(train_mask.sum().item())

        if n_train > 0:
            loss = torch.nn.functional.cross_entropy(
                out[train_mask], y[train_mask], reduction="mean"
            )
            loss_value = float(loss.item())
        else:
            loss = out.sum() * 0.0
            loss_value = 0.0

        metrics = {
            "loss": loss_value,
            "val_acc": compute_micro_f1(out, y, val_mask),
            "test_acc": compute_micro_f1(out, y, test_mask),
            "_n_train": n_train,
        }
        return loss, metrics

    def _train_single_partition_fast_path(
        self, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        optimizer.zero_grad()
        out = self._single_partition_logits()
        y = self.graph.partition_labels(0).to(self.device)
        loss, metrics = self._single_partition_metrics(out, y)
        loss.backward()
        optimizer.step()
        return metrics

    @torch.no_grad()
    def _evaluate_single_partition_fast_path(self) -> Dict[str, float]:
        out = self._single_partition_logits()
        y = self.graph.partition_labels(0).to(self.device)
        _, metrics = self._single_partition_metrics(out, y)
        return metrics

    # ==================================================================
    # Forward Pass
    # ==================================================================

    def _forward_layer(self, layer_id: int) -> None:
        """Forward one layer across assigned partitions with double buffering."""
        pool_size = self.config.pool_size
        pids = self._my_pids

        if not pids:
            return

        # Storage_to_Host: load layer activations into cache
        self._prepare_cache_layer(layer_id)

        # Prologue: prefetch first assigned partition
        self._prepare_cache_partition(layer_id, pids[0])
        self.device_features[layer_id].async_gather(
            pid=pids[0],
            host_buffer=self.host_features[layer_id],
            boundaries=self.graph.boundaries[pids[0]],
            stream=self.streams.h2d[0],
        )

        for i, pid in enumerate(pids):
            pool_idx = i % pool_size

            # Wait for H2D of current partition
            self.device_features[layer_id].h2d_synchronize(
                self.streams.h2d[pool_idx]
            )
            self.streams.compute.wait_stream(self.streams.h2d[pool_idx])
            if self.config.mode == "grinnder":
                self.streams.compute.wait_stream(self.streams.act_h2d[pool_idx])

            with torch.cuda.stream(self.streams.compute):
                # Prefetch next partition (overlap I/O with compute)
                if i < len(pids) - 1 and pool_size > 1:
                    next_pid = pids[i + 1]
                    next_pool = (i + 1) % pool_size
                    self._prepare_cache_partition(layer_id, next_pid)
                    self.device_features[layer_id].async_gather(
                        pid=next_pid,
                        host_buffer=self.host_features[layer_id],
                        boundaries=self.graph.boundaries[next_pid],
                        stream=self.streams.h2d[next_pool],
                    )

                x = self.device_features[layer_id][pid]
                x.requires_grad_(True)

                # saved_tensors_hooks wraps checkpoint(fn, x).
                # checkpoint only saves x. adj loaded inside fn (not passed).
                if self.config.mode == "hongtu":
                    with HongtuCheckpoint(pid, self.device_features[layer_id]):
                        out = checkpoint(
                            self._upoffload_wrapper,
                            layer_id, pid, x,
                            use_reentrant=True,
                        )
                else:  # grinnder
                    with ScatteredCheckpoint(
                        pid,
                        self.device_features[layer_id],
                        self.streams.act_h2d[pool_idx],
                    ):
                        out = checkpoint(
                            self._upoffload_wrapper,
                            layer_id, pid, x,
                            use_reentrant=True,
                        )

                # Keep activation reference
                self.activations[layer_id][pid] = out

                # Output routing decided by cache mode:
                # - lru_layer: D2H to host, evict whole layers only under budget pressure
                # - partition_lru: bypass to NVMe, demand-load partitions
                # - hongtu: D2H to host (no cache)
                use_bypass = (
                    self.cache is not None
                    and self.cache.mode == "partition_lru"
                )
                if use_bypass:
                    self.host_features[layer_id + 1].async_bypass_to_storage(
                        pid, out, self.streams.d2h[pool_idx]
                    )
                else:
                    self.host_features[layer_id + 1].async_fill(
                        pid, out, self.streams.d2h[pool_idx]
                    )

                # Free GPU input features
                self.device_features[layer_id].release(pid)

                # Free previous partition's activation from GPU
                if i > 0:
                    prev_pid = pids[i - 1]
                    prev_pool = (i - 1) % pool_size
                    self.streams.compute.wait_stream(self.streams.d2h[prev_pool])
                    self.host_features[layer_id + 1].d2h_synchronize(
                        self.streams.d2h[prev_pool]
                    )
                    act_prev = self.activations[layer_id][prev_pid]
                    if act_prev is not None:
                        act_prev.untyped_storage().resize_(0)

            if pool_size == 1:
                self.streams.compute.wait_stream(self.streams.d2h[0])
                self.host_features[layer_id + 1].d2h_synchronize(
                    self.streams.d2h[0]
                )
                act_cur = self.activations[layer_id][pid]
                if act_cur is not None:
                    act_cur.untyped_storage().resize_(0)

                if i < len(pids) - 1:
                    next_pid = pids[i + 1]
                    self._prepare_cache_partition(layer_id, next_pid)
                    self.device_features[layer_id].async_gather(
                        pid=next_pid,
                        host_buffer=self.host_features[layer_id],
                        boundaries=self.graph.boundaries[next_pid],
                        stream=self.streams.h2d[0],
                    )

        # Epilogue: free last assigned partition's activation
        if pool_size > 1:
            last_pool = (len(pids) - 1) % pool_size
            self.streams.compute.wait_stream(self.streams.d2h[last_pool])
            self.host_features[layer_id + 1].d2h_synchronize(
                self.streams.d2h[last_pool]
            )
            last_pid = pids[-1]
            act_last = self.activations[layer_id][last_pid]
            if act_last is not None:
                act_last.untyped_storage().resize_(0)

        # Cache: decide what to keep vs flush
        if self.cache is not None:
            self.cache.on_layer_complete(layer_id)

    def _upoffload_wrapper(self, layer_id: int, pid: int, x: Tensor) -> Tensor:
        """Wrap forward_layer with gradient offload hook.

        adj is loaded INSIDE this function (not passed to checkpoint)
        so that checkpoint only saves x, not adj's sparse tensors.
        """
        x = GradOffload.apply(
            x, layer_id, pid,
            self.host_gradients[layer_id] if layer_id > 0 else None,
            self.graph.boundaries[pid],
            self.streams.d2h[pid % self.config.pool_size],
            self.streams.compute,
        )
        # Load adj inside (not passed to checkpoint)
        adj = self._get_adj(pid)
        out = self.model.forward_layer(layer_id, x, adj)
        del adj
        return out

    # ==================================================================
    # Loss Computation
    # ==================================================================

    def _compute_losses(
        self, criterion: torch.nn.Module
    ) -> tuple[List[Tensor], Dict[str, float]]:
        """Compute per-partition losses with double-buffered activation upload.

        Returns:
            losses: List of (pid, loss) tuples for assigned partitions.
            metrics: dict with loss, val_acc, test_acc.
        """
        pool_size = self.config.pool_size
        pids = self._my_pids
        losses: List[Tensor] = []
        pid_to_loss_idx: Dict[int, int] = {}  # pid -> index in losses list
        total_loss = 0.0
        n_train = 0
        n_val_correct = 0
        n_val = 0
        n_test_correct = 0
        n_test = 0

        last_layer = self.model.num_layers - 1

        if not pids:
            return losses, {"loss": 0.0, "val_acc": 0.0, "test_acc": 0.0}

        if self.cache is not None and self.cache.mode == "lru_layer":
            self.cache.ensure_in_host(self.model.num_layers)

        use_bypass = self.cache is not None and self.cache.mode == "partition_lru"

        # Prologue: upload first assigned partition's activation
        first_pid = pids[0]
        if use_bypass:
            self.host_features[-1].storage_to_cpu(first_pid)
        act_first = self.activations[last_layer][first_pid]
        if act_first is not None:
            act_first.untyped_storage().resize_(
                act_first.numel() * act_first.element_size()
            )
        self.host_features[-1].async_upload(
            first_pid, act_first, self.streams.h2d[0]
        )

        for i, pid in enumerate(pids):
            pool_idx = i % pool_size

            self.host_features[-1].h2d_synchronize(self.streams.h2d[pool_idx])
            torch.cuda.current_stream().wait_stream(self.streams.h2d[pool_idx])

            # Prefetch next partition
            if i < len(pids) - 1 and pool_size > 1:
                next_pid = pids[i + 1]
                if use_bypass:
                    self.host_features[-1].storage_to_cpu(next_pid)
                act_next = self.activations[last_layer][next_pid]
                if act_next is not None:
                    act_next.untyped_storage().resize_(
                        act_next.numel() * act_next.element_size()
                    )
                self.host_features[-1].async_upload(
                    next_pid, act_next,
                    self.streams.h2d[(i + 1) % pool_size],
                )

            # Get masks and labels
            y = self.graph.partition_labels(pid).to(self.device)
            train_mask = self.graph.partition_train_mask(pid)
            val_mask = self.graph.partition_val_mask(pid)
            test_mask = self.graph.partition_test_mask(pid)
            out = self.activations[last_layer][pid]

            # Compute loss with sum reduction for partition-invariant gradients.
            # Each partition contributes sum(losses) / total_train_nodes.
            # This ensures identical gradients regardless of partition count.
            n_part_train = train_mask.sum().item()
            if n_part_train > 0:
                loss = torch.nn.functional.cross_entropy(
                    out[train_mask], y[train_mask], reduction="sum"
                )
                total_loss += loss.item()
                n_train += n_part_train
            else:
                # Keep an autograd-connected zero loss for empty train partitions.
                loss = out.sum() * 0.0

            pid_to_loss_idx[pid] = len(losses)
            losses.append(loss)

            # Compute metrics
            with torch.no_grad():
                if val_mask.sum() > 0:
                    n_val_correct += compute_micro_f1(out, y, val_mask) * val_mask.sum().item()
                    n_val += val_mask.sum().item()
                if test_mask.sum() > 0:
                    n_test_correct += compute_micro_f1(out, y, test_mask) * test_mask.sum().item()
                    n_test += test_mask.sum().item()

            # Free activation
            self.activations[last_layer][pid].untyped_storage().resize_(0)
            if use_bypass:
                self.host_features[-1].release(pid)

            if i < len(pids) - 1 and pool_size == 1:
                next_pid = pids[i + 1]
                if use_bypass:
                    self.host_features[-1].storage_to_cpu(next_pid)
                act_next = self.activations[last_layer][next_pid]
                if act_next is not None:
                    act_next.untyped_storage().resize_(
                        act_next.numel() * act_next.element_size()
                    )
                self.host_features[-1].async_upload(
                    next_pid,
                    act_next,
                    self.streams.h2d[0],
                )

        self._pid_to_loss_idx = pid_to_loss_idx

        metrics = {
            "loss": total_loss / max(n_train, 1),
            "val_acc": n_val_correct / max(n_val, 1),
            "test_acc": n_test_correct / max(n_test, 1),
            "_n_train": n_train,
        }
        return losses, metrics

    # ==================================================================
    # Backward Pass
    # ==================================================================

    def _backward_last_layer(self, losses: List[Tensor]) -> None:
        """Backward for last layer: loss.backward() per partition.

        For scattered mode: re-gathers features of layer (L-2) into
        device_features[L-1] for checkpoint recompute during backward.
        """
        layer_id = self.model.num_layers - 1
        pool_size = self.config.pool_size
        pids = self._my_pids

        if not pids:
            return

        cache_layer_id = layer_id
        self._prepare_cache_layer(cache_layer_id)

        if layer_id > 0 and self.host_gradients[layer_id] is not None:
            self.host_gradients[layer_id].initialize_zeros(
                lazy=self._uses_partition_lru()
            )

        # Prologue: re-gather features for first partition's checkpoint recompute
        if self.config.mode == "grinnder" and layer_id > 0:
            self._prepare_cache_partition(cache_layer_id, pids[0])
            self.device_features[layer_id].async_gather(
                pid=pids[0],
                host_buffer=self.host_features[layer_id],
                boundaries=self.graph.boundaries[pids[0]],
                stream=self.streams.act_h2d[0],
            )

        for i, pid in enumerate(pids):
            pool_idx = i % pool_size
            loss_idx = self._pid_to_loss_idx[pid]

            # Wait for re-gather to complete
            if self.config.mode == "grinnder" and layer_id > 0:
                self.device_features[layer_id].h2d_synchronize(
                    self.streams.act_h2d[pool_idx]
                )
            self.streams.compute.wait_stream(self.streams.act_h2d[pool_idx])

            # Prefetch re-gather for NEXT partition
            if (
                i < len(pids) - 1
                and pool_size > 1
                and self.config.mode == "grinnder"
                and layer_id > 0
            ):
                next_pid = pids[i + 1]
                next_pool = (i + 1) % pool_size
                self._prepare_cache_partition(cache_layer_id, next_pid)
                self.device_features[layer_id].async_gather(
                    pid=next_pid,
                    host_buffer=self.host_features[layer_id],
                    boundaries=self.graph.boundaries[next_pid],
                    stream=self.streams.act_h2d[next_pool],
                )

            with torch.cuda.stream(self.streams.compute):
                # Free previous partition's re-gathered features
                if i > 0:
                    prev_pid = pids[i - 1]
                    prev_pool = (i - 1) % pool_size
                    if layer_id > 0 and self.host_gradients[layer_id] is not None:
                        self.host_gradients[layer_id].d2h_synchronize(
                            self.streams.d2h[prev_pool]
                        )
                    df = self.device_features[layer_id]
                    if df.is_allocated(prev_pid):
                        df.release(prev_pid)

                # Backward through loss (triggers checkpoint recompute)
                loss = losses[loss_idx]
                loss.backward(retain_graph=False)
                self.streams.compute.synchronize()
                losses[loss_idx] = None
                act = self.activations[layer_id][pid]
                if act is not None:
                    act.untyped_storage().resize_(0)
                    act.grad = None
                    self.activations[layer_id][pid] = None

            if pool_size == 1:
                if layer_id > 0 and self.host_gradients[layer_id] is not None:
                    self.host_gradients[layer_id].d2h_synchronize(
                        self.streams.d2h[0]
                    )
                if self.device_features[layer_id].is_allocated(pid):
                    self.device_features[layer_id].release(pid)

                if (
                    i < len(pids) - 1
                    and self.config.mode == "grinnder"
                    and layer_id > 0
                ):
                    next_pid = pids[i + 1]
                    self._prepare_cache_partition(cache_layer_id, next_pid)
                    self.device_features[layer_id].async_gather(
                        pid=next_pid,
                        host_buffer=self.host_features[layer_id],
                        boundaries=self.graph.boundaries[next_pid],
                        stream=self.streams.act_h2d[0],
                    )

        # GradOffload scatters ∇A^{L-1} into the host write-back buffer during
        # the checkpoint recompute. The next backward phase uploads that buffer,
        # so all scatters from this phase must be visible first.
        if layer_id > 0 and self.host_gradients[layer_id] is not None:
            for i, _ in enumerate(pids):
                pool_idx = i % pool_size
                self.host_gradients[layer_id].d2h_synchronize(
                    self.streams.d2h[pool_idx]
                )

        if pool_size > 1:
            last_pid = pids[-1]
            if self.device_features[layer_id].is_allocated(last_pid):
                self.device_features[layer_id].release(last_pid)

        if self._uses_partition_lru() and self.cache is not None:
            self.cache.on_backward_layer_complete(layer_id)

    def _backward_layer(self, layer_id: int) -> None:
        """Backward for middle layer: activation.backward(gradient).

        Per partition, uploads 3 things:
          1. ∇A^{l+1}_p (gradient from next layer) from host_gradients[layer_id+1]
          2. A^l_p (activation for recompute) -- via ScatteredCheckpoint
          3. GA^{l-1}_p (regathered input) -- via ScatteredCheckpoint
        And offloads 1 thing:
          4. ∇GA^{l-1}_p scattered to host_gradients[layer_id] via GradOffload.backward

        Host serves as write-back buffer for gradient accumulation.
        """
        pool_size = self.config.pool_size
        pids = self._my_pids
        next_grad_layer = layer_id + 1  # gradient index from next layer
        prefetch_next_feature = layer_id > 0
        prefetch_next_gradient = layer_id > 0

        if not pids:
            return

        self._prepare_cache_layer(layer_id)

        # Initialize gradient write-back buffer for THIS layer
        if self.host_gradients[layer_id] is not None:
            self.host_gradients[layer_id].initialize_zeros(
                lazy=self._uses_partition_lru()
            )

        first_pid = pids[0]

        # Prologue: prefetch features for regathering + upload gradient for first pid
        if self.config.mode == "grinnder":
            self._prepare_cache_partition(layer_id, first_pid)
            self.device_features[layer_id].async_gather(
                pid=first_pid,
                host_buffer=self.host_features[layer_id],
                boundaries=self.graph.boundaries[first_pid],
                stream=self.streams.act_h2d[0],
            )

        # Upload gradient from next layer for first partition
        if self.host_gradients[next_grad_layer] is not None:
            grad_buf = self.device_gradients[next_grad_layer]
            if grad_buf is not None:
                if (
                    self._uses_partition_lru()
                    and not self.host_gradients[next_grad_layer].is_allocated(
                        first_pid
                    )
                ):
                    self._ensure_gradient_partition_resident(
                        next_grad_layer, first_pid
                    )
                grad_buf.allocate(first_pid)
                self.host_gradients[next_grad_layer].async_upload(
                    first_pid, grad_buf[first_pid], self.streams.h2d[0]
                )

        for i, pid in enumerate(pids):
            pool_idx = i % pool_size

            # Wait for H2D of gradient and regathered features
            if self.config.mode == "grinnder":
                self.device_features[layer_id].h2d_synchronize(
                    self.streams.act_h2d[pool_idx]
                )
            self.streams.compute.wait_stream(self.streams.h2d[pool_idx])
            self.streams.compute.wait_stream(self.streams.act_h2d[pool_idx])

            with torch.cuda.stream(self.streams.compute):
                # Prefetch next partition: regather features + upload gradient
                if i < len(pids) - 1 and pool_size > 1:
                    next_pid = pids[i + 1]
                    next_pool = (i + 1) % pool_size

                    if self.config.mode == "grinnder" and prefetch_next_feature:
                        self._prepare_cache_partition(layer_id, next_pid)
                        self.device_features[layer_id].async_gather(
                            pid=next_pid,
                            host_buffer=self.host_features[layer_id],
                            boundaries=self.graph.boundaries[next_pid],
                            stream=self.streams.act_h2d[next_pool],
                        )

                    if (
                        prefetch_next_gradient
                        and self.host_gradients[next_grad_layer] is not None
                    ):
                        grad_buf = self.device_gradients[next_grad_layer]
                        if grad_buf is not None:
                            if (
                                self._uses_partition_lru()
                                and not self.host_gradients[
                                    next_grad_layer
                                ].is_allocated(next_pid)
                            ):
                                self._ensure_gradient_partition_resident(
                                    next_grad_layer, next_pid
                                )
                            grad_buf.allocate(next_pid)
                            self.host_gradients[next_grad_layer].async_upload(
                                next_pid, grad_buf[next_pid],
                                self.streams.h2d[next_pool],
                            )

                # Wait for previous gradient scatter to complete
                if i > 0:
                    prev_pid = pids[i - 1]
                    prev_pool = (i - 1) % pool_size
                    if self.host_gradients[layer_id] is not None:
                        self.host_gradients[layer_id].d2h_synchronize(
                            self.streams.d2h[prev_pool]
                        )
                    self.device_features[layer_id].release(prev_pid)
                    if self.device_gradients[next_grad_layer] is not None:
                        self.device_gradients[next_grad_layer].release(prev_pid)

                # Synchronize gradient upload
                if self.host_gradients[next_grad_layer] is not None:
                    self.host_gradients[next_grad_layer].h2d_synchronize(
                        self.streams.h2d[pool_idx]
                    )
                    if self._uses_partition_lru():
                        self.host_gradients[next_grad_layer].release(pid)

                # Backward: activation.backward(gradient_from_next_layer)
                act = self.activations[layer_id][pid]
                if act is not None and act.grad_fn is not None:
                    grad_buf = self.device_gradients[next_grad_layer]
                    if grad_buf is not None and grad_buf.is_allocated(pid):
                        act.backward(grad_buf[pid], retain_graph=False)
                    else:
                        act.backward(retain_graph=False)

                    self.streams.compute.synchronize()
                    if (
                        self.device_gradients[next_grad_layer] is not None
                        and self.device_gradients[next_grad_layer].is_allocated(pid)
                    ):
                        self.device_gradients[next_grad_layer].release(pid)
                    act.untyped_storage().resize_(0)
                    act.grad = None
                    self.activations[layer_id][pid] = None

                    if (
                        not prefetch_next_feature
                        and pool_size > 1
                        and i < len(pids) - 1
                        and self.config.mode == "grinnder"
                    ):
                        if self.device_features[layer_id].is_allocated(pid):
                            self.device_features[layer_id].release(pid)
                        next_pid = pids[i + 1]
                        next_pool = (i + 1) % pool_size
                        self._prepare_cache_partition(layer_id, next_pid)
                        self.device_features[layer_id].async_gather(
                            pid=next_pid,
                            host_buffer=self.host_features[layer_id],
                            boundaries=self.graph.boundaries[next_pid],
                            stream=self.streams.act_h2d[next_pool],
                        )

                    if (
                        not prefetch_next_gradient
                        and pool_size > 1
                        and i < len(pids) - 1
                        and self.host_gradients[next_grad_layer] is not None
                    ):
                        next_pid = pids[i + 1]
                        next_pool = (i + 1) % pool_size
                        grad_buf = self.device_gradients[next_grad_layer]
                        if grad_buf is not None:
                            if (
                                self._uses_partition_lru()
                                and not self.host_gradients[
                                    next_grad_layer
                                ].is_allocated(next_pid)
                            ):
                                self._ensure_gradient_partition_resident(
                                    next_grad_layer, next_pid
                                )
                            grad_buf.allocate(next_pid)
                            self.host_gradients[next_grad_layer].async_upload(
                                next_pid,
                                grad_buf[next_pid],
                                self.streams.h2d[next_pool],
                            )

            if pool_size == 1:
                if self.host_gradients[layer_id] is not None:
                    self.host_gradients[layer_id].d2h_synchronize(
                        self.streams.d2h[0]
                    )
                if self.device_features[layer_id].is_allocated(pid):
                    self.device_features[layer_id].release(pid)
                if (
                    self.device_gradients[next_grad_layer] is not None
                    and self.device_gradients[next_grad_layer].is_allocated(pid)
                ):
                    self.device_gradients[next_grad_layer].release(pid)

                if i < len(pids) - 1:
                    next_pid = pids[i + 1]

                    if self.config.mode == "grinnder":
                        self._prepare_cache_partition(layer_id, next_pid)
                        self.device_features[layer_id].async_gather(
                            pid=next_pid,
                            host_buffer=self.host_features[layer_id],
                            boundaries=self.graph.boundaries[next_pid],
                            stream=self.streams.act_h2d[0],
                        )

                    if self.host_gradients[next_grad_layer] is not None:
                        grad_buf = self.device_gradients[next_grad_layer]
                        if grad_buf is not None:
                            if (
                                self._uses_partition_lru()
                                and not self.host_gradients[
                                    next_grad_layer
                                ].is_allocated(next_pid)
                            ):
                                self._ensure_gradient_partition_resident(
                                    next_grad_layer, next_pid
                                )
                            grad_buf.allocate(next_pid)
                            self.host_gradients[next_grad_layer].async_upload(
                                next_pid,
                                grad_buf[next_pid],
                                self.streams.h2d[0],
                            )

        # Epilogue: synchronize last scatter + free last partition
        if pool_size > 1:
            last_pid = pids[-1]
            last_pool = (len(pids) - 1) % pool_size
            if self.host_gradients[layer_id] is not None:
                self.host_gradients[layer_id].d2h_synchronize(
                    self.streams.d2h[last_pool]
                )
            self.device_features[layer_id].release(last_pid)
            if self.device_gradients[next_grad_layer] is not None:
                self.device_gradients[next_grad_layer].release(last_pid)

        if self.cache is not None:
            self.cache.on_backward_layer_complete(layer_id)
