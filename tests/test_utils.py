"""Tests for utility functions."""

import torch
from grinnder.utils import compute_micro_f1, fix_seed, get_available_host_memory


def test_fix_seed():
    fix_seed(42)
    a = torch.randn(10)
    fix_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_compute_micro_f1():
    pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    target = torch.tensor([1, 0, 1])
    assert compute_micro_f1(pred, target) == 1.0


def test_compute_micro_f1_with_mask():
    pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    target = torch.tensor([1, 1, 1])  # pred[1] is wrong
    mask = torch.tensor([True, True, True])
    acc = compute_micro_f1(pred, target, mask)
    assert abs(acc - 2.0 / 3.0) < 1e-5


def test_compute_micro_f1_empty():
    pred = torch.tensor([[0.1, 0.9]])
    target = torch.tensor([1])
    mask = torch.tensor([False])
    assert compute_micro_f1(pred, target, mask) == 0.0


def test_available_memory():
    mem = get_available_host_memory()
    assert mem >= 0


def test_default_partitioner_threads_uses_logical_cpus(monkeypatch):
    from grinnder import utils

    monkeypatch.delenv("GRINNDER_PARTITION_THREADS", raising=False)
    monkeypatch.setattr(utils.os, "cpu_count", lambda: 32)
    assert utils.get_default_partitioner_threads() == 32


def test_default_partitioner_threads_env_override(monkeypatch):
    from grinnder.utils import get_default_partitioner_threads

    monkeypatch.setenv("GRINNDER_PARTITION_THREADS", "12")
    assert get_default_partitioner_threads() == 12
