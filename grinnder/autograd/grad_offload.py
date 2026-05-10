"""Gradient offloading via custom autograd functions.

GradOffload is inserted BEFORE forward_layer() in the computation graph.
During backward, it async-scatters the computed activation gradients
from GPU back to host partition buffers via boundaries.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from grinnder.buffer.host import HostBuffer


class GradOffload(torch.autograd.Function):
    """Custom autograd: offload activation gradients during backward.

    Forward: identity (returns input unchanged).
    Backward: async scatter gradient to host_gradients buffer via boundaries.
              Skipped for layer 0 (input features have no gradient to scatter).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        layer_id: int,
        pid: int,
        host_gradients: Optional[HostBuffer],
        boundaries: List[Optional[Tensor]],
        d2h_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
    ) -> Tensor:
        ctx.layer_id = layer_id
        ctx.pid = pid
        ctx.host_gradients = host_gradients
        ctx.boundaries = boundaries
        ctx.d2h_stream = d2h_stream
        ctx.compute_stream = compute_stream
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if ctx.layer_id == 0 or ctx.host_gradients is None:
            # Input features have no previous layer. Stop here so PyTorch does
            # not materialize a large, unused gathered-feature gradient.
            return None, None, None, None, None, None, None

        # Async scatter gradients to host partitions with accumulation
        # grad_output may not be contiguous from autograd — make it so
        grad_output = grad_output.contiguous()
        ctx.d2h_stream.wait_stream(ctx.compute_stream)
        ctx.host_gradients.async_scatter(
            ctx.pid,
            grad_output,
            ctx.boundaries,
            ctx.d2h_stream,
        )

        # Previous-layer backward is driven explicitly from the host gradient
        # buffer after the scatter completes.
        return None, None, None, None, None, None, None
