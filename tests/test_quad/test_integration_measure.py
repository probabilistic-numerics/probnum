"""Test cases for integration measures."""

import numpy as np
import pytest

from probnum import quad


@pytest.mark.parametrize("dim", [1, 7, 13, 50, 356])
def test_gaussian_diagonal_covariance(dim):
    """Check that diagonal covariance matrices are recognised as diagonal."""
    mean = np.full((dim,), 0.0)
    cov = np.eye(dim)
    measure = quad.GaussianMeasure(mean, cov)
    assert measure.diagonal_covariance == True


@pytest.mark.parametrize("dim", [1, 7, 13, 50, 356])
def test_gaussian_mean_format(dim):
    """Check that the mean of Gaussian measure is formatted correctly."""
    mean = np.full((dim,), 1.5)
    cov = np.eye(dim)
    measure = quad.GaussianMeasure(mean, cov)
    assert measure.mean.shape == (dim, 1)
