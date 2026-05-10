"""Multi-GPU partition parallelism.

Each GPU processes a disjoint subset of partitions. All GPUs share
host memory buffers (same node) for cross-partition gradient accumulation.

Synchronization:
  - Vertex activation gradients: scatter-accumulated via shared host buffers.
    Each GPU writes to its partitions' host buffers; boundary gradients
    cross GPU boundaries through the shared host memory.
  - Weight gradients: all-reduced across GPUs BEFORE optimizer.step().
  - Barrier between forward and backward to ensure all partitions' forward
    completes before any GPU starts backward.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from grinnder.config import GriNNderConfig
from grinnder.data.graph import PartitionedGraph
from grinnder.nn.base import GriNNderModel
from grinnder.engine.trainer import Trainer


class PartitionParallel:
    """Multi-GPU wrapper with partition parallelism.

    Each GPU processes a disjoint subset of graph partitions. Host memory
    buffers are shared across GPUs on the same node for gradient scatter.

    Training flow:
      1. Each GPU runs forward on its assigned partitions.
      2. dist.barrier() — all GPUs finish forward.
      3. Each GPU computes loss on its partitions.
      4. Each GPU runs backward on its partitions (gradient scatter to shared host).
      5. dist.barrier() — all GPUs finish backward.
      6. All-reduce weight gradients.
      7. optimizer.step().

    Args:
        model: GriNNderModel (replicated on each GPU).
        graph: PartitionedGraph.
        config: GriNNderConfig.
        rank: Local GPU rank (0-indexed).
        world_size: Total number of GPUs.
    """

    def __init__(
        self,
        model: GriNNderModel,
        graph: PartitionedGraph,
        config: GriNNderConfig,
        rank: int,
        world_size: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.graph = graph

        assert graph.num_parts >= world_size, (
            f"Need at least {world_size} partitions for {world_size} GPUs, "
            f"got {graph.num_parts}"
        )

        # Assign partitions to GPUs (contiguous blocks)
        parts_per_gpu = graph.num_parts // world_size
        remainder = graph.num_parts % world_size
        # Distribute remainder across first GPUs
        if rank < remainder:
            start = rank * (parts_per_gpu + 1)
            end = start + parts_per_gpu + 1
        else:
            start = remainder * (parts_per_gpu + 1) + (rank - remainder) * parts_per_gpu
            end = start + parts_per_gpu
        self.partition_range = (start, end)
        self.my_partitions = list(range(start, end))

        # Per-GPU config
        gpu_config = GriNNderConfig(
            mode=config.mode,
            num_parts=config.num_parts,
            partitioner=config.partitioner,
            partitioner_kwargs=config.partitioner_kwargs,
            storage_dir=f"{config.storage_dir}/gpu{rank}",
            cache_mode=config.cache_mode,
            host_memory_budget_gb=config.host_memory_budget_gb,
            device=f"cuda:{rank}",
            pool_size=config.pool_size,
            num_gpus=config.num_gpus,
        )

        self.model = model.to(f"cuda:{rank}")

        # Create trainer — it manages ALL partitions' host buffers (shared)
        # but only processes this GPU's assigned partitions.
        self.trainer = Trainer(
            model=self.model,
            graph=graph,
            config=gpu_config,
            partition_range=self.partition_range,
        )

    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ) -> Dict[str, float]:
        """Run one training epoch with partition parallelism.

        Flow:
          1. Forward (each GPU on its partitions)
          2. Barrier
          3. Loss + Backward (each GPU on its partitions)
          4. Barrier
          5. All-reduce weight gradients
          6. optimizer.step()
        """
        self.model.train()
        self.trainer.reset_epoch()
        optimizer.zero_grad()

        # Phase 1: Forward (each GPU on its partitions)
        for layer_id in range(self.model.num_layers):
            self.trainer._forward_layer(layer_id)

        # Barrier: ensure all GPUs finish forward before backward
        dist.barrier()

        # Phase 2: Loss computation
        losses, metrics = self.trainer._compute_losses(criterion)

        # Phase 3: Backward
        self.trainer._backward_last_layer(losses)
        for layer_id in reversed(range(self.model.num_layers - 1)):
            self.trainer._backward_layer(layer_id)

        # Barrier: ensure all GPUs finish backward
        dist.barrier()

        # Phase 4: All-reduce weight gradients BEFORE step
        self._allreduce_weight_gradients()

        # Phase 5: Weight update
        optimizer.step()

        return metrics

    def _allreduce_weight_gradients(self) -> None:
        """All-reduce model weight gradients across GPUs."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)

    @staticmethod
    def setup(rank: int, world_size: int, backend: str = "nccl") -> None:
        """Initialize distributed process group."""
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    @staticmethod
    def cleanup() -> None:
        """Destroy distributed process group."""
        dist.destroy_process_group()
