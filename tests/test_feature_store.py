import numpy as np
import torch

from grinnder.data.datasets import NumpyFeatureStore


def test_numpy_feature_store_materializes_only_requested_partition():
    array = np.arange(24, dtype=np.float32).reshape(6, 4)
    store = NumpyFeatureStore(array)

    part = store.partition(1, 4)

    assert torch.equal(part, torch.from_numpy(array[1:4]))
    assert store.resident_nbytes == 0


def test_numpy_feature_store_applies_partition_permutation():
    array = np.arange(24, dtype=np.float32).reshape(6, 4)
    perm = torch.tensor([4, 2, 0, 5, 3, 1])
    store = NumpyFeatureStore(array).with_permutation(perm)

    part = store.partition(1, 4)
    expected = torch.from_numpy(array[[2, 0, 5]])

    assert store.shape == (6, 4)
    assert store.size(1) == 4
    assert torch.equal(part, expected)
