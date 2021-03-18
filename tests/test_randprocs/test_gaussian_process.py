"""Tests for Gaussian processes."""
import numpy as np

from probnum import randprocs, randvars, utils


def test_finite_evaluation_is_normal(gaussian_process: randprocs.GaussianProcess):
    """A Gaussian process evaluated at a finite set of inputs is a Gaussian random
    variable."""
    x = np.random.normal(size=(5,) + utils.as_shape(gaussian_process.input_dim))
    assert isinstance(gaussian_process(x), randvars.Normal)
