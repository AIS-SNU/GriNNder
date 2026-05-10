"""Train GCN or GAT on IGB-Medium (10M nodes, 120M edges) with GriNNder.

This is the real-world benchmark from the paper. With 10M nodes and 1024-dim
features, the intermediate activations exceed GPU memory, requiring GriNNder's
storage offloading.

Usage:
  python examples/train_igb.py --igb_root data/igb_datasets
  python examples/train_igb.py --igb_root data/igb_datasets --mode hongtu
  python examples/train_igb.py --igb_root data/igb_datasets --cache_mode lru_layer --storage_dir /pci5_nvme/grinnder
"""

import argparse
import statistics
import time

import torch

from grinnder import GAT, GCN, GriNNderConfig, Trainer, build_partitioned_graph
from grinnder.data.datasets import load_igb
from grinnder.utils import fix_seed, get_default_partitioner_threads, report_memory


def build_model(args, graph):
    if args.model == "gcn":
        return GCN(
            in_channels=graph.feat_dim,
            hidden_channels=args.hidden,
            out_channels=graph.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            norm=True,
        )
    if args.model == "gat":
        return GAT(
            in_channels=graph.feat_dim,
            hidden_channels=args.hidden,
            out_channels=graph.num_classes,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            norm=True,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Train GCN or GAT on IGB-Medium with GriNNder")
    parser.add_argument("--igb_root", type=str, default="data/igb_datasets",
                        help="Path to igb_datasets directory")
    parser.add_argument("--igb_size", type=str, default="medium",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True,
                        help="Download homogeneous IGB tiny/small/medium if missing")
    parser.add_argument("--confirm_download", action="store_true",
                        help="Skip the interactive prompt for large IGB downloads")
    parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983])
    parser.add_argument("--mode", type=str, default="grinnder", choices=["hongtu", "grinnder"])
    parser.add_argument("--num_parts", type=int, default=64)
    parser.add_argument("--cache_mode", type=str, default="auto",
                        choices=["auto", "lru_layer", "partition_lru"])
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=1, help="GAT attention heads")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage_dir", type=str, default="/pci5_nvme/grinnder")
    parser.add_argument("--runtime_safety_margin_gb", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--partition_cache_dir", type=str, default="data/grinnder_partitions")
    parser.add_argument("--no_save_partition_cache", action="store_true")
    parser.add_argument("--partitioner_threads", type=int, default=get_default_partitioner_threads())
    parser.add_argument("--preprocess_workers", type=int, default=get_default_partitioner_threads())
    parser.add_argument("--materialize_features", action="store_true",
                        help="Load the full IGB feature matrix into RAM instead of using mmap")
    args = parser.parse_args()

    fix_seed(args.seed)
    device = args.device

    # ---- Load dataset ----
    print(f"Loading IGB-{args.igb_size} dataset from {args.igb_root}...")
    t0 = time.time()
    data = load_igb(
        args.igb_root,
        size=args.igb_size,
        num_classes=args.num_classes,
        mmap_features=not args.materialize_features,
        download=args.download,
        confirm_download=args.confirm_download,
    )
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Nodes: {data.num_nodes:,}, Edges: {data.edge_index.size(1):,}, "
          f"Features: {data.x.size(1)}, Classes: {args.num_classes}")
    print(f"  Train: {data.train_mask.sum():,}, Val: {data.val_mask.sum():,}, "
          f"Test: {data.test_mask.sum():,}")
    print(f"  {report_memory('After load')}")

    # ---- Configure ----
    config = GriNNderConfig(
        mode=args.mode,
        num_parts=args.num_parts,
        partitioner="grinnder",
        partitioner_kwargs={
            "capacity": 1.1,
            "beta": 1.0,
            "max_iter": 50,
            "num_threads": args.partitioner_threads,
        },
        preprocess_workers=args.preprocess_workers,
        storage_dir=args.storage_dir,
        cache_mode=args.cache_mode,
        runtime_safety_margin_gb=args.runtime_safety_margin_gb,
        pool_size=args.pool_size,
        device=device,
    )
    print(
        f"\nMode: {config.mode}, Partitions: {config.num_parts}, "
        f"Cache: {args.cache_mode}, Partitioner threads: {args.partitioner_threads}, "
        f"Preprocess workers: {args.preprocess_workers}, "
        f"Pool size: {args.pool_size}, "
        f"Safety margin: {args.runtime_safety_margin_gb:.1f}GB"
    )

    # ---- Partition graph ----
    print("Partitioning...")
    cache_path = None
    if args.partition_cache_dir:
        from pathlib import Path

        cache_path = (
            Path(args.partition_cache_dir)
            / f"igb_{args.igb_size}_grinnder_{args.num_parts}p.pt"
        )
    t0 = time.time()
    graph = build_partitioned_graph(
        edge_index=data.edge_index,
        x=data.x,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        config=config,
        cache_path=cache_path,
        save_cache=not args.no_save_partition_cache,
    )
    t_part = time.time() - t0
    print(f"  Partitioning: {t_part:.1f}s")
    print(f"  Partition sizes: min={min(graph.partition_sizes):,}, "
          f"max={max(graph.partition_sizes):,}, "
          f"avg={sum(graph.partition_sizes)//len(graph.partition_sizes):,}")
    print(f"  {report_memory('After partition')}")

    # Free original metadata container; graph keeps an mmap-backed feature source.
    del data
    torch.cuda.empty_cache()

    # ---- Create model ----
    model = build_model(args, graph).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {args.model.upper()} "
        f"({args.num_layers} layers, {args.hidden} hidden, {n_params:,} params)"
    )

    # ---- Train ----
    trainer = Trainer(model=model, graph=graph, config=config)
    if trainer.cache is not None:
        plan = trainer.cache.memory_plan
        to_gb = lambda nbytes: nbytes / float(1024**3)
        print(
            "Cache plan: "
            f"mode={trainer.cache.mode}, "
            f"remaining={to_gb(plan.remaining_cache_bytes):.2f}GB, "
            f"activation_budget={to_gb(trainer.cache.activation_cache_budget_bytes):.2f}GB, "
            f"all_layer_residency={to_gb(plan.all_layer_residency_bytes):.2f}GB, "
            f"layer_working_set={to_gb(plan.layer_working_set_bytes):.2f}GB, "
            f"safety_margin={to_gb(plan.safety_margin_bytes):.2f}GB"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nTraining for {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Val Acc':>8} | {'Test Acc':>9} | {'Time':>8}")
    print("-" * 56)

    best_val = 0.0
    best_test = 0.0
    epoch_times = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        metrics = trainer.train_epoch(optimizer, criterion)
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        if metrics["val_acc"] > best_val:
            best_val = metrics["val_acc"]
            best_test = metrics["test_acc"]

        print(
            f"{epoch:6d} | {metrics['loss']:8.4f} | {metrics['val_acc']:7.2%} | "
            f"{metrics['test_acc']:8.2%} | {t_epoch:7.1f}s"
        )

    t_total = time.time() - t_start

    print("-" * 56)
    print(
        f"\nTraining complete in {t_total:.1f}s "
        f"({statistics.median(epoch_times):.1f}s median epoch)"
    )
    print(f"Best Val: {best_val:.2%} | Best Test: {best_test:.2%}")
    print(f"\n{report_memory('Final')}")


if __name__ == "__main__":
    main()
