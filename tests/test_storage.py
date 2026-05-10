"""Tests for StorageBackend and StorageTensor."""

import os
import inspect
import tempfile

import torch
import pytest

try:
    import kvikio
    HAS_KVIKIO = True
except ImportError:
    HAS_KVIKIO = False

from grinnder.storage.tensor import StorageTensor


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def backend(tmp_dir):
    pytest.importorskip("kvikio")
    from grinnder.storage.backend import StorageBackend
    return StorageBackend(tmp_dir)


class TestStorageBackend:
    def test_host_write_read(self, backend):
        t = torch.randn(10, 5)
        h = backend.host_write(t, "test1")
        backend.wait(h)
        assert backend.exists("test1")

        out = torch.empty_like(t)
        h = backend.host_read("test1", out)
        backend.wait(h)
        assert torch.allclose(t, out)

    def test_file_management(self, backend):
        t = torch.randn(3, 3)
        h = backend.host_write(t, "f1")
        backend.wait(h)
        assert backend.exists("f1")
        assert backend.file_size("f1") > 0

        backend.remove("f1")
        assert not backend.exists("f1")

    def test_cleanup(self, backend):
        t = torch.randn(5)
        h1 = backend.host_write(t, "a")
        h2 = backend.host_write(t, "b")
        backend.wait(h1)
        backend.wait(h2)
        backend.cleanup()
        assert not backend.exists("a")
        assert not backend.exists("b")


class TestStorageTensorWithBackend:
    """Tests requiring kvikio + io_uring (StorageBackend)."""

    def test_lifecycle(self, backend):
        st = StorageTensor((10, 5), torch.float32, backend, "st_test", pinned=False)
        assert st.location == "host"

        st.host_data.fill_(3.14)

        st.to_storage(async_=False)
        assert st.location == "storage"

        st.to_host(async_=False)
        st.finish_to_host()
        assert st.location == "host"
        assert torch.allclose(st.host_data, torch.full((10, 5), 3.14))


class TestStorageTensorBasic:
    """Tests that don't need StorageBackend."""

    def test_default_host_allocation_is_pageable(self):
        default = inspect.signature(StorageTensor.__init__).parameters["pinned"].default
        assert default is False

        st = StorageTensor((2, 3), torch.float32, None, "default_pageable")
        assert not st.host_data.is_pinned()

    def test_reset(self):
        # Use a dummy backend=None since reset doesn't touch storage
        st = StorageTensor.__new__(StorageTensor)
        st._shape = (5, 3)
        st._dtype = torch.float32
        st._backend = None
        st._file_id = "dummy"
        st._element_size = 4
        st._numel = 15
        st._nbytes = 60
        st._pinned = False
        st._host_tensor = torch.empty(5, 3, dtype=torch.float32)
        st._location = "host"
        st._io_handle = -1

        st.host_data.fill_(1.0)
        st.reset()
        assert (st.host_data == 0).all()
