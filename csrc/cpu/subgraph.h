#pragma once

#include <torch/extension.h>

// Extract 1-hop subgraph with contiguous node relabeling.
//
// Given a CSR graph (rowptr, col, optional value) and seed nodes (idx),
// returns a subgraph containing all seed nodes and their 1-hop neighbors.
// Nodes are relabeled to contiguous [0, N) where:
//   [0, idx.size()) = seed (destination) nodes
//   [idx.size(), N) = boundary (source-only) nodes
//
// If bipartite=true: rowptr has idx.size()+1 entries (destination rows only).
// If bipartite=false: rowptr has N+1 entries (all nodes get rows, boundary
//                     nodes have empty rows).
//
// Returns: (out_rowptr, out_col, out_value_or_none, all_node_ids)
std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
build_subgraph(torch::Tensor rowptr, torch::Tensor col,
               torch::optional<torch::Tensor> optional_value,
               torch::Tensor idx, bool bipartite);
