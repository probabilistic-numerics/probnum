"""Tests for Gaussian processes."""

import pytest

from probnum import backend, randprocs, randvars
from probnum.randprocs import kernels, mean_fns
import tests.utils


def test_mean_not_function_raises_error():
    with pytest.raises(TypeError):
        randprocs.GaussianProcess(
            mean=backend.zeros_like,
            cov=kernels.ExpQuad(input_shape=(1,)),
        )


def test_cov_not_kernel_raises_error():
    """Initializing a GP with a covariance function which is not a kernel raises a
    TypeError."""
    with pytest.raises(TypeError):
        randprocs.GaussianProcess(
            mean=mean_fns.Zero(input_shape=(1,), output_shape=(1,)),
            cov=lambda x0, x1: backend.exp(-backend.abs(x0 - x1)),
        )


def test_mean_kernel_shape_mismatch_raises_error():
    with pytest.raises(ValueError):
        randprocs.GaussianProcess(
            mean=mean_fns.Zero(input_shape=(2,), output_shape=(1,)),
            cov=kernels.ExpQuad(input_shape=(3,)),
        )

    with pytest.raises(ValueError):
        randprocs.GaussianProcess(
            mean=mean_fns.Zero(input_shape=(2,), output_shape=(2,)),
            cov=kernels.ExpQuad(input_shape=(2,)),
        )


def test_mean_wrong_input_shape_raises_error():
    with pytest.raises(ValueError):
        randprocs.GaussianProcess(
            mean=mean_fns.Zero(input_shape=(2, 2), output_shape=(1,)),
            cov=kernels.ExpQuad(input_shape=(2,)),
        )

    with pytest.raises(ValueError):
        randprocs.GaussianProcess(
            mean=mean_fns.Zero(input_shape=(2,), output_shape=(2, 1)),
            cov=kernels.ExpQuad(input_shape=(2,)),
        )


def test_finite_evaluation_is_normal(gaussian_process: randprocs.GaussianProcess):
    """A Gaussian process evaluated at a finite set of inputs is a Gaussian random
    variable."""
    x_shape = (5,) + gaussian_process.input_shape
    x = backend.random.standard_normal(
        seed=tests.utils.random.seed_from_sampling_args(
            base_seed=98998123,
            shape=x_shape,
        ),
        shape=x_shape,
    )
    assert isinstance(gaussian_process(x), randvars.Normal)
