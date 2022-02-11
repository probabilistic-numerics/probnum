"""Tests for Gaussian processes."""

import numpy as np
import pytest

from probnum import randprocs, randvars


def test_no_kernel_covariance_raises_error():
    """Initializing a GP with a covariance function which is not a kernel raises a
    TypeErrror."""
    with pytest.raises(TypeError):
        randprocs.GaussianProcess(mean=np.zeros_like, cov=np.dot)


def test_finite_evaluation_is_normal(gaussian_process: randprocs.GaussianProcess):
    """A Gaussian process evaluated at a finite set of inputs is a Gaussian random
    variable."""
    x = np.random.normal(size=(5,) + gaussian_process.input_shape)
    assert isinstance(gaussian_process(x), randvars.Normal)
