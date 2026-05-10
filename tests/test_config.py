"""Tests for GriNNderConfig."""

import os

from grinnder.config import GriNNderConfig


def test_default_config():
    config = GriNNderConfig()
    default_threads = os.cpu_count() or 1
    assert config.mode == "grinnder"
    assert config.num_parts == 8
    assert config.partitioner == "grinnder"
    assert config.pool_size == 2
    assert config.cache_mode == "auto"
    assert config.storage_dir == "/pci5_nvme/grinnder"
    assert config.preprocess_workers == default_threads
    assert config.partitioner_kwargs["num_threads"] == default_threads


def test_thread_env_override(monkeypatch):
    monkeypatch.setenv("GRINNDER_PARTITION_THREADS", "12")
    config = GriNNderConfig()
    assert config.preprocess_workers == 12
    assert config.partitioner_kwargs["num_threads"] == 12


def test_hongtu_mode():
    config = GriNNderConfig(mode="hongtu")
    assert config.mode == "hongtu"


def test_custom_partitioner_kwargs():
    config = GriNNderConfig(
        partitioner_kwargs={"capacity": 1.2, "beta": 0.5, "max_iter": 100}
    )
    assert config.partitioner_kwargs["capacity"] == 1.2
    assert config.partitioner_kwargs["max_iter"] == 100
