"""Graph partitioning and subgraph construction using grdpart API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from grinnder.config import GriNNderConfig
from grinnder.data.graph import PartitionedGraph
from grinnder.data.datasets import compute_gcn_norm
from grinnder.storage.backend import StorageBackend

PARTITION_CACHE_VERSION = 1


def _load_subgraph_fn():
    """Lazy-load C++ build_subgraph function."""
    try:
        from grinnder._C import build_subgraph
        return build_subgraph
    except ImportError:
        return _build_subgraph_python


def _build_subgraph_python(
    rowptr: Tensor,
    col: Tensor,
    value: Optional[Tensor],
    idx: Tensor,
    bipartite: bool,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """Pure Python fallback for 1-hop subgraph extraction."""
    rowptr_data = rowptr.numpy()
    col_data = col.numpy()
    idx_data = idx.numpy()

    node_map = {}
    boundary_ids = []

    for i, v in enumerate(idx_data):
        node_map[int(v)] = i

    out_rowptr = [0]
    out_col_list = []
    out_val_list = []

    for i, v in enumerate(idx_data):
        start, end = int(rowptr_data[v]), int(rowptr_data[v + 1])
        for j in range(start, end):
            w = int(col_data[j])
            if w not in node_map:
                node_map[w] = len(idx_data) + len(boundary_ids)
                boundary_ids.append(w)
            out_col_list.append(node_map[w])
            if value is not None:
                out_val_list.append(value[j].item())
        out_rowptr.append(len(out_col_list))

    if not bipartite:
        for _ in boundary_ids:
            out_rowptr.append(len(out_col_list))

    out_rowptr_t = torch.tensor(out_rowptr, dtype=torch.long)
    out_col_t = torch.tensor(out_col_list, dtype=torch.long)
    out_val_t = torch.tensor(out_val_list, dtype=value.dtype) if value is not None else None
    all_ids = torch.cat([idx, torch.tensor(boundary_ids, dtype=torch.long)])

    return out_rowptr_t, out_col_t, out_val_t, all_ids


def _edge_index_to_csr(
    edge_index: Tensor, num_nodes: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert COO edge_index to CSR (rowptr, col)."""
    row, col = edge_index[0], edge_index[1]
    # Sort by row
    perm = row.argsort()
    row = row[perm]
    col = col[perm]
    # Build rowptr
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr.scatter_add_(0, row + 1, torch.ones_like(row))
    rowptr = rowptr.cumsum(0)
    return rowptr, col, perm


def _reorder_graph_payload(
    x: Any,
    y: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    perm: Tensor,
) -> Tuple[Any, Tensor, Tensor, Tensor, Tensor]:
    if hasattr(x, "with_permutation"):
        x = x.with_permutation(perm)
    else:
        x = x[perm]
    return x, y[perm], train_mask[perm], val_mask[perm], test_mask[perm]


def _partition_cache_matches(
    cache: Dict[str, Any],
    *,
    num_nodes: int,
    num_edges: int,
    num_parts: int,
    partitioner: str,
) -> bool:
    return (
        cache.get("version") == PARTITION_CACHE_VERSION
        and cache.get("num_nodes") == num_nodes
        and cache.get("input_num_edges") == num_edges
        and cache.get("num_parts") == num_parts
        and cache.get("partitioner") == partitioner
    )


def _load_partition_cache(path: Union[str, Path]) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _graph_from_partition_cache(
    cache: Dict[str, Any],
    x: Any,
    y: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    backend: Optional[StorageBackend],
) -> PartitionedGraph:
    perm = cache["perm"]
    x, y, train_mask, val_mask, test_mask = _reorder_graph_payload(
        x, y, train_mask, val_mask, test_mask, perm
    )

    adj_csr_list = cache["adj_csr"]
    if backend is not None:
        for pid, (rp, c, v) in enumerate(adj_csr_list):
            backend.host_write(rp, f"adj_rowptr_p{pid}", async_=False)
            backend.host_write(c, f"adj_col_p{pid}", async_=False)
            if v is not None:
                backend.host_write(v, f"adj_val_p{pid}", async_=False)

    return PartitionedGraph(
        num_nodes=cache["num_nodes"],
        num_edges=cache["num_edges"],
        num_parts=cache["num_parts"],
        partition_sizes=cache["partition_sizes"],
        adj_csr=adj_csr_list,
        boundaries=cache["boundaries"],
        expanded_sizes=cache["expanded_sizes"],
        features=x,
        labels=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        perm=perm,
        ptr=cache["ptr"],
    )


def _write_partition_cache(path: Union[str, Path], cache: Dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, tmp_path)
    tmp_path.replace(cache_path)


def _build_partition_subgraph(
    pid: int,
    ptr: Tensor,
    rowptr: Tensor,
    col: Tensor,
    edge_weight: Tensor,
    num_parts: int,
    build_subgraph,
) -> Tuple[int, Tuple[Tensor, Tensor, Optional[Tensor]], List[Optional[Tensor]], int, int]:
    start = int(ptr[pid].item())
    end = int(ptr[pid + 1].item())
    batch_size = end - start
    seed_nodes = torch.arange(start, end, dtype=torch.long)

    # Extract 1-hop subgraph (bipartite)
    sub_rowptr, sub_col, sub_value, n_id = build_subgraph(
        rowptr, col, edge_weight, seed_nodes, True
    )

    # Build boundaries with in-partition vertex ordering.
    other_n_id = n_id[batch_size:]
    boundaries: List[Optional[Tensor]] = [None] * num_parts

    new_order_indices = []
    for src_pid in range(num_parts):
        if src_pid == pid:
            continue
        src_start = int(ptr[src_pid].item())
        src_end = int(ptr[src_pid + 1].item())
        mask = (other_n_id >= src_start) & (other_n_id < src_end)
        if mask.any():
            idx_in_other = mask.nonzero(as_tuple=True)[0]
            new_order_indices.append(idx_in_other)
            boundaries[src_pid] = other_n_id[idx_in_other] - src_start

    # Remap adjacency columns: discovery order -> partition order
    n_boundary = len(other_n_id)
    if n_boundary > 0 and len(new_order_indices) > 0:
        reorder = torch.cat(new_order_indices)
        inv_reorder = torch.empty(n_boundary, dtype=torch.long)
        for new_pos, old_pos in enumerate(reorder.tolist()):
            inv_reorder[old_pos] = new_pos

        new_col = sub_col.clone()
        boundary_mask = sub_col >= batch_size
        old_boundary_col = sub_col[boundary_mask] - batch_size
        new_col[boundary_mask] = inv_reorder[old_boundary_col] + batch_size
        sub_col = new_col

    expanded_size = batch_size + n_boundary
    return pid, (sub_rowptr, sub_col, sub_value), boundaries, expanded_size, batch_size


def build_partitioned_graph(
    edge_index: Tensor,
    x: Any,
    y: Tensor,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    config: GriNNderConfig,
    backend: Optional[StorageBackend] = None,
    cache_path: Optional[Union[str, Path]] = None,
    save_cache: bool = True,
) -> PartitionedGraph:
    """Partition a graph and prepare for GriNNder training.

    Steps:
      1. Call grdpart to get partition assignments.
      2. Reorder nodes by partition (via result.perm).
      3. Compute GCN normalization on FULL graph (before splitting).
      4. For each partition: extract 1-hop bipartite subgraph.
      5. Build boundary lists.
      6. In-partition vertex ordering for sequential access.
      7. Store per-partition adjacencies on NVMe.

    Args:
        edge_index: [2, E] COO edge index.
        x: [N, F] node features.
        y: [N] node labels.
        train_mask, val_mask, test_mask: [N] boolean masks.
        config: GriNNder configuration.
        backend: StorageBackend for NVMe (None = keep in RAM).

    Returns:
        PartitionedGraph ready for training.
    """
    from grdpart import GrinnderPartitioner, SpinnerPartitioner

    num_nodes = x.size(0)
    input_num_edges = int(edge_index.size(1))

    if cache_path is not None and Path(cache_path).is_file():
        cache = _load_partition_cache(cache_path)
        if _partition_cache_matches(
            cache,
            num_nodes=num_nodes,
            num_edges=input_num_edges,
            num_parts=config.num_parts,
            partitioner=config.partitioner,
        ):
            print(f"Loading partitioned graph cache from {cache_path}")
            return _graph_from_partition_cache(
                cache, x, y, train_mask, val_mask, test_mask, backend
            )
        print(f"Ignoring incompatible partitioned graph cache at {cache_path}")

    # Step 1: Partition
    if config.partitioner == "grinnder":
        partitioner = GrinnderPartitioner(
            num_parts=config.num_parts, **config.partitioner_kwargs
        )
    elif config.partitioner == "spinner":
        partitioner = SpinnerPartitioner(
            num_parts=config.num_parts, **config.partitioner_kwargs
        )
    else:
        raise ValueError(
            f"Unsupported partitioner {config.partitioner!r}. "
            "Supported partitioners are 'grinnder' and 'spinner'."
        )

    result = partitioner.partition(edge_index, num_nodes=num_nodes)

    # Step 2: Reorder nodes by partition
    perm = result.perm
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, dtype=torch.long)

    x, y, train_mask, val_mask, test_mask = _reorder_graph_payload(
        x, y, train_mask, val_mask, test_mask, perm
    )

    # Remap edge_index through inverse permutation
    edge_index = inv_perm[edge_index]

    # Step 3: GCN normalization on FULL reordered graph
    edge_weight = compute_gcn_norm(edge_index, num_nodes, add_self_loops=True)

    # Add self-loops to edge_index (matching compute_gcn_norm)
    from torch_geometric.utils import add_self_loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Convert to CSR
    rowptr, col, sort_perm = _edge_index_to_csr(edge_index, num_nodes)
    edge_weight = edge_weight[sort_perm]

    # Step 4-6: Per-partition subgraph extraction
    build_subgraph = _load_subgraph_fn()
    ptr = result.ptr

    adj_csr_list: List[Optional[Tuple[Tensor, Tensor, Optional[Tensor]]]] = [
        None
    ] * config.num_parts
    boundaries_list: List[Optional[List[Optional[Tensor]]]] = [None] * config.num_parts
    expanded_sizes = [0] * config.num_parts
    partition_sizes = [0] * config.num_parts
    preprocess_workers = max(1, int(config.preprocess_workers))
    worker_args = (
        ptr,
        rowptr,
        col,
        edge_weight,
        config.num_parts,
        build_subgraph,
    )

    if preprocess_workers == 1 or config.num_parts == 1:
        results = [
            _build_partition_subgraph(pid, *worker_args)
            for pid in range(config.num_parts)
        ]
    else:
        max_workers = min(preprocess_workers, config.num_parts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_build_partition_subgraph, pid, *worker_args)
                for pid in range(config.num_parts)
            ]
            results = [future.result() for future in futures]

    for pid, adj_csr, boundaries, expanded_size, batch_size in results:
        adj_csr_list[pid] = adj_csr
        boundaries_list[pid] = boundaries
        expanded_sizes[pid] = expanded_size
        partition_sizes[pid] = batch_size

    adj_csr_list_final = [
        adj for adj in adj_csr_list if adj is not None
    ]
    boundaries_list_final = [
        boundaries for boundaries in boundaries_list if boundaries is not None
    ]

    # Step 7: Optionally store adjacencies on NVMe
    if backend is not None:
        for pid in range(config.num_parts):
            rp, c, v = adj_csr_list_final[pid]
            backend.host_write(rp, f"adj_rowptr_p{pid}", async_=False)
            backend.host_write(c, f"adj_col_p{pid}", async_=False)
            if v is not None:
                backend.host_write(v, f"adj_val_p{pid}", async_=False)

    graph_num_edges = int(edge_index.size(1))
    if cache_path is not None and save_cache:
        _write_partition_cache(
            cache_path,
            {
                "version": PARTITION_CACHE_VERSION,
                "num_nodes": num_nodes,
                "input_num_edges": input_num_edges,
                "num_edges": graph_num_edges,
                "num_parts": config.num_parts,
                "partitioner": config.partitioner,
                "partition_sizes": partition_sizes,
                "adj_csr": adj_csr_list_final,
                "boundaries": boundaries_list_final,
                "expanded_sizes": expanded_sizes,
                "perm": perm,
                "ptr": ptr,
            },
        )

    return PartitionedGraph(
        num_nodes=num_nodes,
        num_edges=graph_num_edges,
        num_parts=config.num_parts,
        partition_sizes=partition_sizes,
        adj_csr=adj_csr_list_final,
        boundaries=boundaries_list_final,
        expanded_sizes=expanded_sizes,
        features=x,
        labels=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        perm=perm,
        ptr=ptr,
    )
