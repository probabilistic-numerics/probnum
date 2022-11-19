"""Tests for the basic array object and associated functions."""
import numpy as np

from probnum import Backend

import pytest

try:
    import jax.numpy as jnp
except ImportError as e:
    pass

try:
    import torch
except ImportError as e:
    pass


@pytest.mark.skipif_backend(Backend.NUMPY)
@pytest.mark.skipif_backend(Backend.TORCH)
def test_jax_ndarray_module_is_not_updated():
    assert jnp.ndarray.__module__ != "probnum.backend"


@pytest.mark.skipif_backend(Backend.JAX)
@pytest.mark.skipif_backend(Backend.TORCH)
def test_numpy_ndarray_module_is_not_updated():
    assert np.ndarray.__module__ != "probnum.backend"


@pytest.mark.skipif_backend(Backend.JAX)
@pytest.mark.skipif_backend(Backend.NUMPY)
def test_torch_tensor_module_is_not_updated():
    assert torch.Tensor.__module__ != "probnum.backend"
