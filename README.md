# GriNNder [![MLSys 2026](https://img.shields.io/badge/MLSys-2026-blue)](#citation) [![Paper](https://img.shields.io/badge/OpenReview-8SNPzGRldN-blue)](https://openreview.net/forum?id=8SNPzGRldN) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[MLSys 2026] GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading**



GriNNder trains full-graph GNNs on graphs whose activations/gradients exceed GPU memory by
coordinating GPU memory, host RAM, and NVMe storage.
The active open-source
surface focuses on OGBN/IGB datasets, GCN/GAT models, and the GriNNder/Spinner
partitioners from [`grdpart`](https://github.com/AIS-SNU/grdpart).


We refactored the paper version codebase for modularity, usability, and maintainability.
Thus, the training performance of the current codebase is faster than the paper version, and the code is more accessible for future contributions.

## Setup

The recommended path is the setup script. It creates the conda environment,
installs PyTorch/PyG/OGB/kvikio/grdpart, installs pytest for repository checks,
builds bundled liburing, and installs the GriNNder extension.

```bash
bash scripts/setup_env.sh grinnder cu124
conda activate grinnder
```

For a portable extension build across several GPU generations, set
`TORCH_CUDA_ARCH_LIST` before running the setup script, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0+PTX" bash scripts/setup_env.sh grinnder cu124
```

The benchmark scripts assume the environment is named `grinnder`. Override with
`ENV_NAME=...` if you use a different name.

Main dependencies:

| Component | Purpose |
|-----------|---------|
| PyTorch + PyG | GNN layers and sparse operations |
| OGB | OGBN dataset downloads and official train/valid/test splits |
| kvikio | GPUDirect Storage path when available |
| liburing 2.8 | io_uring async storage backend |
| grdpart | Lightweight GriNNder and Spinner graph partitioners |

`grdpart` is a separate GitHub repository. METIS support is planned for a
later release.

## Usage

Run a full IGB-medium training job:

```bash
conda run --no-capture-output -n grinnder python examples/train_igb.py \
  --igb_root data/igb_datasets \
  --igb_size medium \
  --model gcn \
  --hidden 256 \
  --num_parts 16 \
  --storage_dir /pci5_nvme/grinnder \
  --confirm_download
```

Use `--model gat --heads 1 --num_parts 64` for the GAT path on a 24GB GPU.
Storage offloading defaults to `/pci5_nvme/grinnder`; change it with
`--storage_dir` or `STORAGE_DIR`.
If a run exceeds GPU memory, increase `--num_parts` to reduce the per-partition
working set size.

Programmatic usage:

```python
import torch
from grinnder import GCN, GriNNderConfig, Trainer, build_partitioned_graph, load_dataset

data = load_dataset("ogbn-products", root="data")
config = GriNNderConfig(
    mode="grinnder",
    num_parts=4,
    partitioner="grinnder",
    cache_mode="auto",
    storage_dir="/pci5_nvme/grinnder",
)

graph = build_partitioned_graph(
    data.edge_index,
    data.x,
    data.y,
    data.train_mask,
    data.val_mask,
    data.test_mask,
    config,
)

model = GCN(graph.feat_dim, 256, graph.num_classes, num_layers=2, norm=True).cuda()
trainer = Trainer(model, graph, config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for _ in range(10):
    metrics = trainer.train_epoch(optimizer, torch.nn.CrossEntropyLoss())
```

### Cache Policy

`cache_mode="auto"` is the default. It resolves in this order:

| Resolved mode | Meaning |
|---------------|---------|
| `lru_layer` | Keep or evict whole activation/gradient layers. If host memory is large enough, no eviction occurs and the cost is mainly cache-management overhead. |
| `partition_lru` | Evict dependency partitions when even one full layer plus working buffers cannot fit. This is intended for larger graphs such as 100M-scale workloads. |

The auto decision accounts for fixed resident graph metadata, labels, masks,
transfer buffers, gradient write-back buffers, and a runtime safety margin.
Long-lived host cache tensors are pageable by default so the cache can release
host memory cleanly under LRU eviction. Pinned CPU memory should be reserved for
bounded transfer or staging buffers when explicitly requested.

## Reproducible Results

Use the README wrapper to download full datasets when missing and run the public
matrix:

```bash
bash scripts/reproduce_readme_results.sh
```

This runs:

| Dataset | Model | Partitions | Notes |
|---------|-------|------------|-------|
| OGBN-Products | GCN hidden 256 | 1, 2, 4 | Full dataset, GriNNder partitioner |
| IGB-medium | GCN hidden 256 | 16 | Cached partitions under `data/grinnder_partitions` |
| IGB-medium | GAT hidden 256, head 1 | 64 | 32 parts OOMs on this 24GB A5000; 64 parts fits |

The scripts create `docs/` when needed and write fresh result artifacts there:

| File | Contents |
|------|----------|
| `docs/products_grinnder_gcn_h256_e10.csv` | Products GCN 1/2/4 partition rows |
| `docs/igbm_gcn_h256_16p_e10.csv` | IGB-medium GCN row |
| `docs/igbm_gat_h256_heads1_64p_e10.csv` | IGB-medium GAT row |
| `docs/reproducible_results.md` | Combined README result summary |

The IGB script uses `/pci5_nvme/grinnder` by default for temporary offloaded
storage. Partition preprocessing is cached by default, so later IGB runs skip
the expensive partition/cache build and usually start from
`Loading partitioned graph cache ...`.
Partitioner and preprocessing worker counts default to all visible CPU threads;
set `GRINNDER_PARTITION_THREADS` or pass the script flags to override this.


The reproduction scripts set dropout to 0 to isolate partitioning equivalence
across partition counts. Tune model hyperparameters separately for best task
accuracy.

### Current Data

Environment: single NVIDIA RTX A5000 24GB, 128GB host RAM, 10 epochs,
GriNNder auto cache mode.

| Dataset | Model | Hidden | Heads | Parts | Cache | Time/epoch (s) | Time/epoch (min) | 10-epoch Val Acc (%) | 10-epoch Test Acc (%) | Peak CUDA (GB) |
|---------|-------|--------|-------|-------|-------|----------------|------------------|----------------------|-----------------------|----------------|
| OGBN-Products | GCN | 256 | | 1 | `lru_layer` | 1.549 | 0.026 | 84.94 | 69.07 | 16.398 |
| OGBN-Products | GCN | 256 | | 2 | `lru_layer` | 5.132 | 0.086 | 84.94 | 69.07 | 10.121 |
| OGBN-Products | GCN | 256 | | 4 | `lru_layer` | 6.733 | 0.112 | 84.94 | 69.07 | 6.168 |
| IGB-medium | GCN | 256 | | 16 | `lru_layer` | 102.381 | 1.706 | 65.12 | 65.20 | 21.081 |
| IGB-medium | GAT | 256 | 1 | 64 | `lru_layer` | 103.550 | 1.726 | 64.39 | 64.50 | 14.810 |

The active GriNNder partitioner uses the GriNNder partitioning path in
`grdpart` (not METIS).
OGBN-Products results show that the accuracy is unaffected by partitioning, and
the full-graph training algorithm is intact. IGB-medium results show that
GriNNder can train full-graph GCN/GAT models with 256 hidden dimensions on a
10M-node graph with 1024D initial features on a 24GB GPU by offloading to NVMe
storage.

## Future Support

- METIS partitioner support through a dedicated wrapper around MT-METIS.
- Benchmark scripts/results for larger hidden dimensions and graphs.
- High-performance multi-GPU training with the adoption of state-of-the-art distributed full-graph training framework (i.e., [`GraNNDis`](https://dl.acm.org/doi/10.1145/3656019.3676892)).
- Sync batch normalization for multiple partitions.
- Priority: METIS support (- July 2026) -> Benchmarks for larger models/datasets (- Sep 2026) -> others (- Dec 2026).

The `grinnder/distributed/` module is kept as a template for future multi-GPU
work. It is not advertised as the current production path.

## Notes

- GriNNder currently requires enough host memory for at least one full layer's
  gradient/write-back working set. If that does not fit, the current write-back
  path cannot run correctly.
- GPUDirect Storage is optional. Without `nvidia-fs`, kvikio falls back to a
  compatible path that is functionally correct but slower for bypass I/O.
- When writing codebase, we referenced [`PyGAS/GNNAutoScale`](https://github.com/rusty1s/pyg_autoscale) a lot for coding style and modularity. Thanks to the authors for the open-source codebase. Please cite this repository/paper if you also reference our codebase for your work. We set MIT license following the PyGAS codebase, so you can use our codebase for your work under the same license.

## Development

Active tests should be run from the repository root:

```bash
conda run --no-capture-output -n grinnder python -m pytest tests -q
```

For CUDA tests that print native partitioner logs, verbose unbuffered mode is
more reliable:

```bash
conda run --no-capture-output -n grinnder python -m pytest tests/test_gradient_identity.py -vv -s
```

## Citation

Paper: https://openreview.net/forum?id=8SNPzGRldN

```bibtex
@inproceedings{song2026grinnder,
  title={{GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading}},
  author={Song, Jaeyong and Park, Seongyeon and Jang, Hongsun and Jung, Jaewon and Lim, Hunseong and Hong, Junguk and Lee, Jinho},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  year={2026}
}
```

## Contact

Questions: `aisys.grads@gmail.com`, open an issue on GitHub, or contact to the first author.

Accelerated Intelligent Systems Lab (AISys) is affiliated with ECE, Seoul
National University.

## License

MIT License. See [LICENSE](LICENSE).
