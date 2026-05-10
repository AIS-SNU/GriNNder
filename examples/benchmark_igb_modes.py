"""Benchmark IGB homogeneous datasets with GriNNder auto by default.

Usage:
  conda run -n grinnder python examples/benchmark_igb_modes.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import shutil
import statistics
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from grinnder import GAT, GCN, GriNNderConfig, Trainer, build_partitioned_graph
from grinnder.data.datasets import load_igb
from grinnder.utils import fix_seed, get_default_partitioner_threads


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


def cuda_cleanup(device: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    except RuntimeError:
        pass


def max_state_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return max((a[k].detach().cpu() - b[k].detach().cpu()).abs().max().item() for k in a)


def build_model(model_name: str, graph, args, hidden: int) -> torch.nn.Module:
    if model_name == "gcn":
        return GCN(
            graph.feat_dim,
            hidden,
            graph.num_classes,
            num_layers=args.num_layers,
            dropout=0.0,
            norm=True,
        )
    if model_name == "gat":
        return GAT(
            graph.feat_dim,
            hidden,
            graph.num_classes,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=0.0,
            norm=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def gb(nbytes: int) -> float:
    return nbytes / float(1024**3)


def build_graph(data, args, num_parts: int):
    cache_path = None
    if args.partition_cache_dir:
        cache_path = (
            Path(args.partition_cache_dir)
            / f"igb_{args.igb_size}_grinnder_{num_parts}p.pt"
        )
    config = GriNNderConfig(
        mode="grinnder",
        num_parts=num_parts,
        partitioner="grinnder",
        partitioner_kwargs={
            "capacity": args.partitioner_capacity,
            "beta": args.partitioner_beta,
            "max_iter": args.partitioner_max_iter,
            "num_threads": args.partitioner_threads,
        },
        preprocess_workers=args.preprocess_workers,
        cache_mode="auto",
        device=args.device,
        storage_dir=tempfile.mkdtemp(prefix=f"grinnder_igb_graph_{num_parts}p_"),
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
        cache_path=cache_path,
        save_cache=not args.no_save_partition_cache,
    )
    return graph, time.perf_counter() - start


def run_trial(
    graph,
    initial_state: Dict[str, torch.Tensor],
    args,
    model_name: str,
    hidden: int,
    num_parts: int,
    mode_spec: ModeSpec,
) -> Dict[str, object]:
    storage_base = Path(args.storage_dir)
    storage_base.mkdir(parents=True, exist_ok=True)
    storage_path = Path(
        tempfile.mkdtemp(
            prefix=f"grinnder_igb_{model_name}_{hidden}_{mode_spec.name}_{num_parts}p_",
            dir=storage_base,
        )
    )
    trainer = None
    optimizer = None
    criterion = None
    model = None
    try:
        config = GriNNderConfig(
            mode=mode_spec.mode,
            num_parts=num_parts,
            partitioner="grinnder",
            cache_mode=mode_spec.cache_mode,
            device=args.device,
            pool_size=args.pool_size,
            storage_dir=str(storage_path),
            host_memory_budget_gb=args.host_memory_budget_gb,
            runtime_safety_margin_gb=args.runtime_safety_margin_gb,
        )
        model = build_model(model_name, graph, args, hidden).to(args.device)
        model.load_state_dict(copy.deepcopy(initial_state))
        print("    trainer setup start", flush=True)
        trainer = Trainer(model, graph, config)
        resolved_cache_mode = trainer.cache.mode if trainer.cache is not None else "n/a"
        if trainer.cache is not None:
            plan = trainer.cache.memory_plan
            print(
                "    cache memory plan "
                f"total_budget_gb={gb(plan.total_budget_bytes):.2f} "
                f"fixed_resident_gb={gb(plan.fixed_resident_bytes):.2f} "
                f"safety_margin_gb={gb(plan.safety_margin_bytes):.2f} "
                f"remaining_cache_gb={gb(plan.remaining_cache_bytes):.2f} "
                f"activation_cache_budget_gb={gb(trainer.cache.activation_cache_budget_bytes):.2f} "
                f"all_layer_residency_gb={gb(plan.all_layer_residency_bytes):.2f} "
                f"layer_working_set_gb={gb(plan.layer_working_set_bytes):.2f} "
                f"largest_dependency_gb={gb(plan.largest_partition_dependency_bytes):.2f}",
                flush=True,
            )
        print(
            "    trial setup "
            f"storage_dir={storage_path} resolved_cache_mode={resolved_cache_mode}",
            flush=True,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)

        epoch_times: List[float] = []
        train_metrics: Dict[str, float] = {}
        for epoch in range(1, args.epochs + 1):
            print(
                f"    epoch {epoch}/{args.epochs} start "
                f"model={model_name} hidden={hidden} parts={num_parts} "
                f"mode={mode_spec.name}",
                flush=True,
            )
            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            train_metrics = trainer.train_epoch(optimizer, criterion)
            torch.cuda.synchronize(args.device)
            epoch_s = time.perf_counter() - start
            epoch_times.append(epoch_s)
            print(
                f"    epoch {epoch}/{args.epochs} done "
                f"time_s={epoch_s:.3f} train_loss={train_metrics.get('loss', 0.0):.6f} "
                f"val_acc={train_metrics.get('val_acc', 0.0):.6f} "
                f"test_acc={train_metrics.get('test_acc', 0.0):.6f}",
                flush=True,
            )

        print("    eval start", flush=True)
        start = time.perf_counter()
        eval_metrics = trainer.evaluate(criterion)
        torch.cuda.synchronize(args.device)
        eval_time = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated(args.device) / (1024**3)
        print(
            f"    eval done time_s={eval_time:.3f} "
            f"val_acc={eval_metrics['val_acc']:.6f} "
            f"test_acc={eval_metrics['test_acc']:.6f} "
            f"peak_cuda_gb={peak_gb:.3f}",
            flush=True,
        )
        result = {
            "status": "ok",
            "error": "",
            "dataset": f"igb-{args.igb_size}",
            "model": model_name,
            "hidden": hidden,
            "heads": args.heads if model_name == "gat" else "",
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
        if not args.keep_storage:
            shutil.rmtree(storage_path, ignore_errors=True)
        gc.collect()
        cuda_cleanup(args.device)

    return result


def error_result(
    args,
    model_name: str,
    hidden: int,
    num_parts: int,
    mode_spec: ModeSpec,
    error: BaseException,
) -> Dict[str, object]:
    return {
        "dataset": f"igb-{args.igb_size}",
        "model": model_name,
        "hidden": hidden,
        "heads": args.heads if model_name == "gat" else "",
        "parts": num_parts,
        "mode": mode_spec.name,
        "cache_mode": mode_spec.cache_mode,
        "resolved_cache_mode": "",
        "status": "oom" if _is_oom(error) else "error",
        "epochs": args.epochs,
        "min_epoch_s": "",
        "median_epoch_s": "",
        "mean_epoch_s": "",
        "eval_s": "",
        "loss": "",
        "val_acc": "",
        "test_acc": "",
        "train_loss_last": "",
        "peak_cuda_gb": "",
        "state_max_diff_vs_hongtu": "",
        "error": str(error).replace("\n", " ")[:500],
    }


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in text


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "hidden",
        "heads",
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
        "state_max_diff_vs_hongtu",
        "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IGB mode matrix")
    parser.add_argument("--igb_root", type=str, default="data/igb_datasets")
    parser.add_argument("--igb_size", choices=["tiny", "small", "medium", "large"], default="medium")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm_download", action="store_true")
    parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983])
    parser.add_argument("--parts", type=parse_int_list, default=[8, 16])
    parser.add_argument("--hiddens", type=parse_int_list, default=[256])
    parser.add_argument("--models", type=parse_str_list, default=["gcn"])
    parser.add_argument(
        "--modes",
        type=parse_str_list,
        default=DEFAULT_MODES,
        help=(
            "Comma-separated mode names. Default: grinnder_auto. "
            "HongTu is available as an opt-in diagnostic mode."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--storage_dir", type=str, default="/pci5_nvme/grinnder")
    parser.add_argument("--keep_storage", action="store_true")
    parser.add_argument("--host_memory_budget_gb", type=float, default=0.0)
    parser.add_argument("--runtime_safety_margin_gb", type=float, default=8.0)
    parser.add_argument("--materialize_features", action="store_true")
    parser.add_argument("--partition_cache_dir", type=str, default="data/grinnder_partitions")
    parser.add_argument("--no_save_partition_cache", action="store_true")
    parser.add_argument("--partitioner_threads", type=int, default=get_default_partitioner_threads())
    parser.add_argument("--preprocess_workers", type=int, default=get_default_partitioner_threads())
    parser.add_argument("--partitioner_capacity", type=float, default=1.1)
    parser.add_argument("--partitioner_beta", type=float, default=1.0)
    parser.add_argument("--partitioner_max_iter", type=int, default=50)
    parser.add_argument("--output_csv", type=Path, default=Path("docs/igbm_modes_results.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    invalid_models = sorted(set(args.models) - {"gcn", "gat"})
    if invalid_models:
        raise ValueError(f"Unsupported models: {invalid_models}")

    mode_specs = selected_modes(args.modes)
    fix_seed(args.seed)
    data = load_igb(
        args.igb_root,
        size=args.igb_size,
        num_classes=args.num_classes,
        mmap_features=not args.materialize_features,
        download=args.download,
        confirm_download=args.confirm_download,
    )
    print(
        "IGB mode benchmark: "
        f"dataset=igb-{args.igb_size}, nodes={data.num_nodes}, "
        f"edges={data.edge_index.size(1)}, epochs={args.epochs}, "
        f"parts={args.parts}, models={args.models}, hiddens={args.hiddens}, "
        f"modes={args.modes}, "
        f"partitioner_threads={args.partitioner_threads}, "
        f"preprocess_workers={args.preprocess_workers}"
    )

    rows: List[Dict[str, object]] = []
    references: Dict[Tuple[str, int, int], Dict[str, torch.Tensor]] = {}

    for num_parts in args.parts:
        print(f"\nBuilding IGB {num_parts}-part graph...")
        graph, partition_time = build_graph(data, args, num_parts)
        print(f"  partition_time_s={partition_time:.6f}")

        initial_states: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
        for model_name in args.models:
            for hidden in args.hiddens:
                seed_model = build_model(model_name, graph, args, hidden)
                initial_states[(model_name, hidden)] = copy.deepcopy(
                    seed_model.state_dict()
                )
                del seed_model

        for model_name in args.models:
            for hidden in args.hiddens:
                for mode_spec in mode_specs:
                    print(
                        f"Running model={model_name}, hidden={hidden}, "
                        f"parts={num_parts}, mode={mode_spec.name}..."
                    )
                    try:
                        row = run_trial(
                            graph,
                            initial_states[(model_name, hidden)],
                            args,
                            model_name,
                            hidden,
                            num_parts,
                            mode_spec,
                        )
                        ref_key = (model_name, hidden, num_parts)
                        if mode_spec.name == "hongtu" and row["status"] == "ok":
                            references[ref_key] = row["state_dict"]
                            row["state_max_diff_vs_hongtu"] = 0.0
                        elif row["status"] == "ok" and ref_key in references:
                            row["state_max_diff_vs_hongtu"] = max_state_diff(
                                references[ref_key],
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
                        row = error_result(args, model_name, hidden, num_parts, mode_spec, exc)
                        print(f"  {row['status']}: {row['error']}")
                        traceback.clear_frames(exc.__traceback__)
                        gc.collect()
                        cuda_cleanup(args.device)
                    row["partition_time_s"] = partition_time
                    rows.append(row)
                    write_csv(args.output_csv, rows)

        del graph
        gc.collect()
        cuda_cleanup(args.device)

    write_csv(args.output_csv, rows)
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
