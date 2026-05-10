"""Benchmark OGBN-Products across partitioners, HongTu, and GriNNder auto.

Runs the public partitioner/parts matrix:
  partitioners = grinnder, spinner
  parts = 1, 2, 4
  modes = GriNNder auto by default; HongTu is available as an opt-in mode

The benchmark uses LayerNorm and dropout=0 so accuracy and final state should
remain partition-invariant for successful trials.

Usage:
  conda run -n grinnder python examples/benchmark_products_modes.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import statistics
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from grinnder import GCN, GriNNderConfig, Trainer, build_partitioned_graph, load_dataset
from grinnder.utils import fix_seed


@dataclass(frozen=True)
class ModeSpec:
    name: str
    mode: str
    cache_mode: str


MODES = (
    ModeSpec("hongtu", "hongtu", "auto"),
    ModeSpec("grinnder_auto", "grinnder", "auto"),
)
DEFAULT_MODES = ["grinnder_auto"]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def selected_modes(names: Optional[List[str]]) -> List[ModeSpec]:
    if names is None:
        return list(MODES)
    by_name = {mode.name: mode for mode in MODES}
    invalid = sorted(set(names) - set(by_name))
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}")
    return [by_name[name] for name in names]


def _sample_from_mask(
    mask: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    idx = mask.nonzero(as_tuple=True)[0]
    if count >= idx.numel():
        return idx
    perm = torch.randperm(idx.numel(), generator=generator)[:count]
    return idx[perm]


def induced_products_subset(data: Data, sample_nodes: int, seed: int) -> Data:
    if sample_nodes <= 0 or sample_nodes >= data.num_nodes:
        return data

    generator = torch.Generator().manual_seed(seed)
    n_train = max(1, int(sample_nodes * 0.70))
    n_val = max(1, int(sample_nodes * 0.15))
    n_test = max(1, sample_nodes - n_train - n_val)

    subset = torch.cat(
        [
            _sample_from_mask(data.train_mask, n_train, generator),
            _sample_from_mask(data.val_mask, n_val, generator),
            _sample_from_mask(data.test_mask, n_test, generator),
        ]
    ).unique()

    if subset.numel() < sample_nodes:
        mask = torch.ones(data.num_nodes, dtype=torch.bool)
        mask[subset] = False
        remaining = mask.nonzero(as_tuple=True)[0]
        fill_count = min(sample_nodes - subset.numel(), remaining.numel())
        fill_perm = torch.randperm(remaining.numel(), generator=generator)[:fill_count]
        subset = torch.cat([subset, remaining[fill_perm]]).unique()

    subset = subset[torch.randperm(subset.numel(), generator=generator)]
    edge_index, _ = subgraph(
        subset,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )

    return Data(
        x=data.x[subset].contiguous(),
        y=data.y[subset].contiguous(),
        edge_index=edge_index.contiguous(),
        train_mask=data.train_mask[subset].contiguous(),
        val_mask=data.val_mask[subset].contiguous(),
        test_mask=data.test_mask[subset].contiguous(),
        num_nodes=int(subset.numel()),
    )


def max_state_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return max((a[k].detach().cpu() - b[k].detach().cpu()).abs().max().item() for k in a)


def cuda_cleanup(device: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    except RuntimeError:
        # A failed CUDA kernel poisons the process context. Preserve the
        # original benchmark row and let the next process-level run continue.
        pass


def build_graph(data: Data, args, partitioner: str, num_parts: int):
    config = GriNNderConfig(
        mode="grinnder",
        num_parts=num_parts,
        partitioner=partitioner,
        cache_mode="auto",
        device=args.device,
        storage_dir=tempfile.mkdtemp(prefix=f"grinnder_products_graph_{num_parts}p_"),
    )
    fix_seed(args.seed)
    start = time.perf_counter()
    graph = build_partitioned_graph(
        data.edge_index,
        data.x,
        data.y,
        data.train_mask,
        data.val_mask,
        data.test_mask,
        config,
    )
    return graph, time.perf_counter() - start


def run_trial(
    graph,
    initial_state: Dict[str, torch.Tensor],
    args,
    partitioner: str,
    num_parts: int,
    mode_spec: ModeSpec,
) -> Dict[str, object]:
    storage_ctx = tempfile.TemporaryDirectory(
        prefix=f"grinnder_products_{mode_spec.name}_{num_parts}p_"
    )
    trainer = None
    optimizer = None
    criterion = None
    model = None
    try:
        config = GriNNderConfig(
            mode=mode_spec.mode,
            num_parts=num_parts,
            partitioner=partitioner,
            cache_mode=mode_spec.cache_mode,
            device=args.device,
            storage_dir=storage_ctx.name,
            host_memory_budget_gb=args.host_memory_budget_gb,
        )
        model = GCN(
            graph.feat_dim,
            args.hidden,
            graph.num_classes,
            num_layers=args.num_layers,
            dropout=0.0,
            norm=True,
        ).to(args.device)
        model.load_state_dict(copy.deepcopy(initial_state))
        trainer = Trainer(model, graph, config)
        resolved_cache_mode = trainer.cache.mode if trainer.cache is not None else "n/a"
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize()

        epoch_times: List[float] = []
        train_metrics: Dict[str, float] = {}
        for _ in range(args.epochs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            train_metrics = trainer.train_epoch(optimizer, criterion)
            torch.cuda.synchronize()
            epoch_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        eval_metrics = trainer.evaluate(criterion)
        torch.cuda.synchronize()
        eval_time = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated(args.device) / (1024**3)
        result = {
            "status": "ok",
            "error": "",
            "partitioner": partitioner,
            "parts": num_parts,
            "mode": mode_spec.name,
            "cache_mode": mode_spec.cache_mode,
            "resolved_cache_mode": resolved_cache_mode,
            "epochs": args.epochs,
            "min_epoch_s": min(epoch_times),
            "median_epoch_s": statistics.median(epoch_times),
            "mean_epoch_s": statistics.mean(epoch_times),
            "eval_s": eval_time,
            "loss": eval_metrics["loss"],
            "val_acc": eval_metrics["val_acc"],
            "test_acc": eval_metrics["test_acc"],
            "train_loss_last": train_metrics.get("loss", 0.0),
            "peak_cuda_gb": peak_gb,
            "state_dict": copy.deepcopy(model.state_dict()),
        }
    finally:
        del trainer, optimizer, criterion, model
        storage_ctx.cleanup()
        gc.collect()
        cuda_cleanup(args.device)

    return result


def oom_result(num_parts: int, mode_spec: ModeSpec, args, error: BaseException) -> Dict[str, object]:
    return {
        "status": "oom" if _is_oom(error) else "error",
        "error": str(error).replace("\n", " ")[:500],
        "partitioner": args.partitioner,
        "parts": num_parts,
        "mode": mode_spec.name,
        "cache_mode": mode_spec.cache_mode,
        "resolved_cache_mode": "",
        "epochs": args.epochs,
        "median_epoch_s": "",
        "mean_epoch_s": "",
        "eval_s": "",
        "loss": "",
        "val_acc": "",
        "test_acc": "",
        "train_loss_last": "",
        "peak_cuda_gb": "",
        "state_dict": None,
    }


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in text


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "partitioner",
        "parts",
        "mode",
        "cache_mode",
        "resolved_cache_mode",
        "status",
        "epochs",
        "min_epoch_s",
        "median_epoch_s",
        "mean_epoch_s",
        "eval_s",
        "loss",
        "val_acc",
        "test_acc",
        "train_loss_last",
        "peak_cuda_gb",
        "partition_time_s",
        "state_max_diff_vs_ref",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OGBN-Products mode matrix")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--sample_nodes", type=int, default=0)
    parser.add_argument("--parts", type=parse_int_list, default=[1, 2, 4])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--partitioner", choices=["grinnder", "spinner"], default="grinnder")
    parser.add_argument("--partitioners", type=parse_str_list, default=None)
    parser.add_argument(
        "--modes",
        type=parse_str_list,
        default=DEFAULT_MODES,
        help=(
            "Comma-separated mode names. Default: grinnder_auto. "
            "HongTu is available as an opt-in diagnostic mode."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host_memory_budget_gb", type=float, default=0.0)
    parser.add_argument("--output_csv", type=Path, default=Path("docs/products_modes_results.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    partitioners = args.partitioners or [args.partitioner]
    invalid = sorted(set(partitioners) - {"grinnder", "spinner"})
    if invalid:
        raise ValueError(f"Unsupported partitioners: {invalid}")
    mode_specs = selected_modes(args.modes)

    fix_seed(args.seed)
    full_data = load_dataset("ogbn-products", root=args.root)
    data = induced_products_subset(full_data, args.sample_nodes, args.seed)
    print(
        "OGBN-Products mode benchmark: "
        f"full_nodes={full_data.num_nodes}, sample_nodes={data.num_nodes}, "
        f"sample_edges={data.num_edges}, epochs={args.epochs}, "
        f"partitioners={','.join(partitioners)}, "
        f"modes={','.join(mode.name for mode in mode_specs)}"
    )

    seed_model = GCN(
        data.x.size(1),
        args.hidden,
        int(data.y.max().item()) + 1,
        num_layers=args.num_layers,
        dropout=0.0,
        norm=True,
    )
    initial_state = copy.deepcopy(seed_model.state_dict())

    rows: List[Dict[str, object]] = []
    reference_state: Optional[Dict[str, torch.Tensor]] = None

    for partitioner in partitioners:
        args.partitioner = partitioner
        for num_parts in args.parts:
            print(f"\nBuilding {partitioner} {num_parts}-part graph...")
            graph, partition_time = build_graph(data, args, partitioner, num_parts)
            print(f"  partition_time_s={partition_time:.6f}")

            for mode_spec in mode_specs:
                print(
                    f"Running partitioner={partitioner}, "
                    f"parts={num_parts}, mode={mode_spec.name}..."
                )
                try:
                    row = run_trial(
                        graph, initial_state, args, partitioner, num_parts, mode_spec
                    )
                    if reference_state is None and row["status"] == "ok":
                        reference_state = row["state_dict"]
                        row["state_max_diff_vs_ref"] = 0.0
                    elif row["status"] == "ok" and reference_state is not None:
                        row["state_max_diff_vs_ref"] = max_state_diff(
                            reference_state,
                            row["state_dict"],
                        )
                    row.pop("state_dict", None)
                    print(
                        "  ok "
                        f"median_epoch_s={row['median_epoch_s']:.6f} "
                        f"val_acc={row['val_acc']:.6f} "
                        f"test_acc={row['test_acc']:.6f} "
                        f"peak_cuda_gb={row['peak_cuda_gb']:.3f}"
                    )
                except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as exc:
                    row = oom_result(num_parts, mode_spec, args, exc)
                    print(f"  {row['status']}: {row['error']}")
                    traceback.clear_frames(exc.__traceback__)
                    gc.collect()
                    cuda_cleanup(args.device)
                row["partition_time_s"] = partition_time
                rows.append(row)
                write_csv(args.output_csv, rows)

    write_csv(args.output_csv, rows)
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
