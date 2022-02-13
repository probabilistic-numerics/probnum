"""Tests for Gaussian processes."""

import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.randprocs import kernels, mean_fns


def test_mean_not_function_raises_error():
    with pytest.raises(TypeError):
        randprocs.GaussianProcess(
            mean=lambda x: np.zeros_like(x),
            cov=kernels.ExpQuad(input_shape=(1,)),
        )


def test_cov_not_kernel_raises_error():
    """Initializing a GP with a covariance function which is not a kernel raises a
    TypeError."""
    with pytest.raises(TypeError):
        randprocs.GaussianProcess(
            mean=mean_fns.Zero(input_shape=(1,), output_shape=(1,)), cov=np.dot
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
    x = np.random.normal(size=(5,) + gaussian_process.input_shape)
    assert isinstance(gaussian_process(x), randvars.Normal)
