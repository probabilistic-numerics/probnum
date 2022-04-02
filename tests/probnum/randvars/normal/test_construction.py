"""Test the construction of Normal random variables."""
import pytest
from pytest_cases import parametrize
from probnum.backend.typing import ShapeType
import tests.utils
from probnum import backend, randvars


@parametrize(shape=[(), (3,), (2, 2)])
def test_mean_cov_shape_mismatch(shape: ShapeType):
    seed = tests.utils.random.seed_from_sampling_args(base_seed=54784, shape=shape)
    mean = backend.random.standard_normal(seed=seed, shape=shape)
    cov = backend.eye(10)

    with pytest.raises(ValueError):
        randvars.Normal(mean=mean, cov=cov)
