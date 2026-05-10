#include "subgraph.h"

#include <unordered_map>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
build_subgraph(torch::Tensor rowptr, torch::Tensor col,
               torch::optional<torch::Tensor> optional_value,
               torch::Tensor idx, bool bipartite) {

  AT_ASSERTM(!rowptr.is_cuda(), "rowptr must be a CPU tensor");
  AT_ASSERTM(!col.is_cuda(), "col must be a CPU tensor");
  AT_ASSERTM(!idx.is_cuda(), "idx must be a CPU tensor");
  if (optional_value.has_value()) {
    AT_ASSERTM(!optional_value.value().is_cuda(), "value must be a CPU tensor");
    AT_ASSERTM(optional_value.value().dim() == 1, "value must be 1D");
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  // Map: original node ID -> relabeled contiguous ID
  std::unordered_map<int64_t, int64_t> node_map;
  // Boundary nodes discovered during traversal
  std::vector<int64_t> boundary_ids;

  // Seed nodes get IDs [0, idx.size())
  auto out_rowptr = torch::empty(idx.numel() + 1, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();

  out_rowptr_data[0] = 0;
  int64_t total_edges = 0;
  for (int64_t i = 0; i < idx.numel(); i++) {
    int64_t v = idx_data[i];
    node_map[v] = i;
    total_edges += rowptr_data[v + 1] - rowptr_data[v];
    out_rowptr_data[i + 1] = total_edges;
  }

  auto out_col = torch::empty(total_edges, col.options());
  auto out_col_data = out_col.data_ptr<int64_t>();

  torch::optional<torch::Tensor> out_value = torch::nullopt;

  if (optional_value.has_value()) {
    out_value = torch::empty(total_edges, optional_value.value().options());

    AT_DISPATCH_ALL_TYPES(
        optional_value.value().scalar_type(), "build_subgraph_valued", [&] {
          auto value_data = optional_value.value().data_ptr<scalar_t>();
          auto out_value_data = out_value.value().data_ptr<scalar_t>();

          int64_t offset = 0;
          for (int64_t i = 0; i < idx.numel(); i++) {
            int64_t v = idx_data[i];
            int64_t row_start = rowptr_data[v];
            int64_t row_end = rowptr_data[v + 1];

            for (int64_t j = row_start; j < row_end; j++) {
              int64_t w = col_data[j];
              auto it = node_map.find(w);
              if (it == node_map.end()) {
                int64_t new_id = idx.numel() + boundary_ids.size();
                node_map[w] = new_id;
                boundary_ids.push_back(w);
                out_col_data[offset] = new_id;
              } else {
                out_col_data[offset] = it->second;
              }
              out_value_data[offset] = value_data[j];
              offset++;
            }
          }
        });
  } else {
    int64_t offset = 0;
    for (int64_t i = 0; i < idx.numel(); i++) {
      int64_t v = idx_data[i];
      int64_t row_start = rowptr_data[v];
      int64_t row_end = rowptr_data[v + 1];

      for (int64_t j = row_start; j < row_end; j++) {
        int64_t w = col_data[j];
        auto it = node_map.find(w);
        if (it == node_map.end()) {
          int64_t new_id = idx.numel() + boundary_ids.size();
          node_map[w] = new_id;
          boundary_ids.push_back(w);
          out_col_data[offset] = new_id;
        } else {
          out_col_data[offset] = it->second;
        }
        offset++;
      }
    }
  }

  // If not bipartite, extend rowptr for boundary nodes (empty rows)
  if (!bipartite) {
    out_rowptr = torch::cat(
        {out_rowptr,
         torch::full({(int64_t)boundary_ids.size()}, out_col.numel(),
                     rowptr.options())});
  }

  // Build full node ID tensor: [seed_nodes | boundary_nodes]
  auto all_ids = torch::cat(
      {idx, torch::from_blob(boundary_ids.data(),
                             {(int64_t)boundary_ids.size()}, idx.options())
               .clone()});

  return std::make_tuple(out_rowptr, out_col, out_value, all_ids);
}
