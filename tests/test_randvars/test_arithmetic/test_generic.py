"""Tests for generic random variable arithmetic."""

import numpy as np
from numpy.typing import DTypeLike

from probnum import backend, randvars
from probnum.backend.typing import ShapeLike

import pytest


@pytest.mark.parametrize("shape,dtype", [((5,), np.single), ((2, 3), np.double)])
def test_generic_randvar_dtype_shape_inference(shape: ShapeLike, dtype: DTypeLike):
    x = randvars.RandomVariable(
        shape=shape,
        dtype=dtype,
        sample=lambda seed, sample_shape: backend.zeros(sample_shape + shape),
    )
    y = np.array(5.0)
    z = x + y
    assert z.dtype == backend.promote_types(dtype, y.dtype)
    assert z.shape == shape
