"""Storage I/O abstraction for NVMe offloading."""

from grinnder.storage.backend import StorageBackend
from grinnder.storage.tensor import StorageTensor

__all__ = ["StorageBackend", "StorageTensor"]
