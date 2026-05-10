"""Dataset loading utilities (PyG compatible)."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tarfile
from typing import Optional, Union
import urllib.request as urlrequest

import numpy as np
import torch
from torch import Tensor


GB = float(1 << 30)

IGB_DATASET_URLS = {
    "homogeneous": {
        "tiny": "https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_tiny.tar.gz",
        "small": "https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_small.tar.gz",
        "medium": "https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_medium.tar.gz",
    }
}

IGB_MD5SUMS = {
    "homogeneous": {
        "tiny": "34856534da55419b316d620e2d5b21be",
        "small": "6781c699723529902ace0a95cafe6fe4",
        "medium": "4640df4ceee46851fd18c0a44ddcc622",
    }
}


@contextmanager
def _torch_load_with_weights_only_disabled():
    """Temporarily keep PyG/OGB processed-data loading compatible with torch 2.6.

    OGB calls ``torch.load(path)`` for processed PyG datasets. PyTorch 2.6 uses
    ``weights_only=True`` by default, which rejects PyG Data pickles. The OGB
    processed file is produced locally from the official OGB archive during
    dataset preparation, so this override is scoped to OGB dataset construction.
    """
    original_load = torch.load

    def load_with_weights_only_disabled(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_weights_only_disabled
    try:
        yield
    finally:
        torch.load = original_load


class NumpyFeatureStore:
    """Mmap-friendly feature matrix wrapper.

    The store preserves tensor-like ``shape`` and ``size`` access while delaying
    materialization until a partition slice is requested. If a permutation is
    attached, slices are interpreted in reordered graph space and translated
    back to the original numpy rows.
    """

    def __init__(
        self,
        array: "np.ndarray",
        perm: Optional[Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.array = array
        self.dtype = dtype
        self._perm = None if perm is None else perm.detach().cpu().numpy()
        n_rows = array.shape[0] if self._perm is None else int(self._perm.shape[0])
        self.shape = (n_rows, int(array.shape[1]))

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def resident_nbytes(self) -> int:
        return 0

    def with_permutation(self, perm: Tensor) -> "NumpyFeatureStore":
        return NumpyFeatureStore(self.array, perm=perm, dtype=self.dtype)

    def partition(self, start: int, end: int) -> Tensor:
        return self[slice(start, end)]

    def __getitem__(self, index) -> Tensor:
        if self._perm is not None:
            index = self._perm[index]
        data = np.array(self.array[index], copy=True)
        tensor = torch.from_numpy(data)
        return tensor.to(dtype=self.dtype)


def _igb_label_filename(num_classes: int) -> str:
    if num_classes == 19:
        return "node_label_19.npy"
    if num_classes == 2983:
        return "node_label_2K.npy"
    raise ValueError("IGB num_classes must be 19 or 2983")


def _igb_processed_dir(root: Union[str, Path], size: str) -> Path:
    return Path(root) / size / "processed"


def _igb_required_paths(root: Union[str, Path], size: str, num_classes: int) -> list[Path]:
    proc = _igb_processed_dir(root, size)
    return [
        proc / "paper" / "node_feat.npy",
        proc / "paper" / _igb_label_filename(num_classes),
        proc / "paper__cites__paper" / "edge_index.npy",
    ]


def _has_igb_dataset(root: Union[str, Path], size: str, num_classes: int) -> bool:
    return all(path.is_file() for path in _igb_required_paths(root, size, num_classes))


def _confirm_large_download(url: str, confirm_download: bool) -> bool:
    if confirm_download or os.environ.get("SKIP_USER_PROMPT", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    with urlrequest.urlopen(url) as response:
        size_gb = int(response.info().get("Content-Length", 0)) / GB
    if size_gb <= 1:
        return True
    answer = input(f"This will download {size_gb:.2f}GB. Will you proceed? (y/N) ")
    return answer.lower() == "y"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlrequest.urlopen(url) as response, destination.open("wb") as out:
        total = int(response.info().get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        last_report_gb = -1
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            report_gb = int(downloaded / GB)
            if report_gb != last_report_gb:
                last_report_gb = report_gb
                if total:
                    print(
                        f"  downloaded {downloaded / GB:.2f}/{total / GB:.2f} GB",
                        flush=True,
                    )
                else:
                    print(f"  downloaded {downloaded / GB:.2f} GB", flush=True)


def _check_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"IGB download checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in IGB archive: {member.name}")
        tar.extractall(root)


def download_igb(
    root: Union[str, Path],
    size: str = "medium",
    dataset_type: str = "homogeneous",
    confirm_download: bool = False,
) -> None:
    """Download and extract the official homogeneous IGB dataset archive."""
    if dataset_type not in IGB_DATASET_URLS or size not in IGB_DATASET_URLS[dataset_type]:
        raise ValueError(
            "Automatic IGB download currently supports homogeneous "
            "tiny/small/medium. Use the official IGB scripts for larger releases."
        )

    root_path = Path(root)
    url = IGB_DATASET_URLS[dataset_type][size]
    archive = root_path / f"igb_{dataset_type}_{size}.tar.gz"
    if not archive.exists():
        if not _confirm_large_download(url, confirm_download):
            raise RuntimeError("IGB download declined by user")
        print(f"Downloading IGB {dataset_type} {size} to {archive}...")
        _download_file(url, archive)
    else:
        print(f"Using existing IGB archive {archive}")

    print("Verifying IGB archive checksum...")
    _check_md5(archive, IGB_MD5SUMS[dataset_type][size])
    print(f"Extracting IGB {dataset_type} {size} under {root_path}...")
    _safe_extract_tar(archive, root_path)
    archive.unlink(missing_ok=True)


def load_igb(
    root: str,
    size: str = "medium",
    num_classes: int = 19,
    mmap_features: bool = True,
    download: bool = False,
    confirm_download: bool = False,
) -> "torch_geometric.data.Data":
    """Load IGB (Illinois Graph Benchmark) homogeneous paper-cites-paper graph.

    See: https://github.com/IllinoisGraphBenchmark/IGB-Datasets

    Args:
        root: Path to igb_datasets directory (contains medium/, large/, etc.).
        size: Dataset size ('medium' = 10M nodes, 'large' = 100M nodes).
        num_classes: Number of label classes (19 or 2983).
        mmap_features: Keep node features mmap-backed and materialize only
            partition slices during preprocessing.
        download: Download the official homogeneous IGB archive if files are
            missing. Automatic download supports tiny/small/medium.
        confirm_download: Skip the interactive large-download confirmation.

    Returns:
        PyG Data object with x, y, edge_index, train/val/test masks.
    """
    from torch_geometric.data import Data

    if download and not _has_igb_dataset(root, size, num_classes):
        download_igb(root, size=size, confirm_download=confirm_download)

    proc = _igb_processed_dir(root, size)
    if not proc.is_dir():
        raise FileNotFoundError(
            f"IGB data not found at {proc}. Pass download=True to fetch "
            "homogeneous tiny/small/medium automatically."
        )
    missing = [path for path in _igb_required_paths(root, size, num_classes) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"IGB data is incomplete. Missing: {missing_text}")

    # Features: keep mmap-backed by default to avoid loading the full matrix.
    feat_path = proc / "paper" / "node_feat.npy"
    feat = np.load(feat_path, mmap_mode="r" if mmap_features else None)
    num_nodes = feat.shape[0]
    feat_dim = feat.shape[1]
    print(f"  IGB-{size}: {num_nodes:,} nodes, {feat_dim} features, {num_classes} classes")

    # Labels
    labels = np.load(proc / "paper" / _igb_label_filename(num_classes))

    # Edges (paper cites paper)
    edges = np.load(proc / "paper__cites__paper" / "edge_index.npy")
    print(f"  Edges: {edges.shape[0]:,}")

    # Convert metadata to tensors. Features stay lazy unless explicitly requested.
    if mmap_features:
        x = NumpyFeatureStore(feat)
    else:
        x = torch.from_numpy(np.asarray(feat)).float()
    y = torch.from_numpy(labels).long()
    edge_index = torch.from_numpy(edges.T.copy()).long().contiguous()

    # Train/val/test split: 80/10/10
    n_train = int(num_nodes * 0.8)
    n_val = int(num_nodes * 0.1)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True

    return Data(
        x=x, y=y, edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=num_nodes,
    )


def _load_ogb_node_dataset(name: str, root: str):
    from ogb.nodeproppred import PygNodePropPredDataset

    with _torch_load_with_weights_only_disabled():
        return PygNodePropPredDataset(name=name, root=root)


def load_dataset(name: str, root: str = "data") -> "torch_geometric.data.Data":
    """Load an OGBN node-classification dataset by name.

    Returns:
        PyG Data object with edge_index, x, y, train_mask, val_mask, test_mask.
    """
    normalized = name.lower()
    if not normalized.startswith("ogbn-"):
        raise ValueError(
            f"Unknown dataset: {name}. Active loaders support OGBN datasets "
            "through load_dataset() and IGB through load_igb()."
        )

    ogb_name = "ogbn-papers100M" if normalized == "ogbn-papers100m" else normalized
    dataset = _load_ogb_node_dataset(name=ogb_name, root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    data.train_mask = _idx_to_mask(split_idx["train"], data.num_nodes)
    data.val_mask = _idx_to_mask(split_idx["valid"], data.num_nodes)
    data.test_mask = _idx_to_mask(split_idx["test"], data.num_nodes)
    data.y = data.y.squeeze(-1)
    return data


def compute_gcn_norm(
    edge_index: Tensor, num_nodes: int, add_self_loops: bool = True
) -> Tensor:
    """Compute GCN normalization D^{-1/2} A D^{-1/2} on the FULL graph.

    IMPORTANT: Must be computed BEFORE partitioning. The normalization
    coefficients are stored as edge values and carried into per-partition
    adjacencies. Do NOT recompute per-subgraph.

    Args:
        edge_index: [2, num_edges] COO format.
        num_nodes: Total number of nodes.
        add_self_loops: Whether to add self-loops before normalization.

    Returns:
        edge_weight: [num_edges] normalization coefficients.
    """
    from torch_geometric.utils import add_self_loops as _add_self_loops, degree

    if add_self_loops:
        edge_index, _ = _add_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = degree(col, num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_weight


def _idx_to_mask(idx: Tensor, num_nodes: int) -> Tensor:
    """Convert index tensor to boolean mask."""
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask
