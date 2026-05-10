#pragma once

#include <torch/extension.h>

// D2H: GPU -> CPU host tensor (on current CUDA stream, via D2H thread pool)
void d2h_copy_async(torch::Tensor src, torch::Tensor dst);

// H2D: CPU host tensor -> GPU (on current CUDA stream, via H2D thread pool)
void h2d_copy_async(torch::Tensor src, torch::Tensor dst);

// Gather: multiple host partitions -> one GPU tensor (via H2D thread pool)
// Layout: [intra(srcs[pid]) | boundary_from_p0 | boundary_from_p1 | ...]
void gather_partitions(int pid, std::vector<torch::Tensor> srcs,
                       torch::Tensor dst,
                       std::vector<torch::Tensor> boundaries);

// Scatter: one GPU tensor -> multiple host partitions with accumulation
// Layout matches gather. Accumulates into dst partitions.
// Synchronizes the stream before accumulation.
void scatter_partitions(int pid, torch::Tensor src,
                        std::vector<torch::Tensor> dsts,
                        std::vector<torch::Tensor> boundaries);

// Thread pool synchronization
void h2d_synchronize();
void d2h_synchronize();
