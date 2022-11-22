"""Test the construction of Normal random variables."""
from probnum import backend, randvars
from probnum.backend.typing import ShapeType

import pytest
from pytest_cases import parametrize
import tests.utils


@parametrize(shape=[(), (3,), (2, 2)])
def test_mean_cov_shape_mismatch(shape: ShapeType):
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=54784, shape=shape
    )
    mean = backend.random.standard_normal(rng_state, shape=shape)
    cov = backend.eye(10)

    with pytest.raises(ValueError):
        randvars.Normal(mean=mean, cov=cov)
