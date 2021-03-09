"""Test cases for integration measures."""

import numpy as np
import pytest

from probnum import quad


def test_gaussian_diagonal_covariance(input_dim: int):
    """Check that diagonal covariance matrices are recognised as diagonal."""
    mean = np.full((input_dim,), 0.0)
    cov = np.eye(input_dim)
    measure = quad.GaussianMeasure(mean, cov)
    assert measure.diagonal_covariance == True


@pytest.mark.parametrize("input_dim_non_diagonal", [2, 10, 100])
def test_gaussian_non_diagonal_covariance(input_dim_non_diagonal):
    """Check that non-diagonal covariance matrices are recognised as non-diagonal."""
    mean = np.full((input_dim_non_diagonal,), 0.0)
    cov = np.eye(input_dim_non_diagonal)
    cov[0, 1] = 1.5
    measure = quad.GaussianMeasure(mean, cov)
    assert measure.diagonal_covariance == False


@pytest.mark.parametrize("mean", [0, np.array([0]), np.array([[0]])])
@pytest.mark.parametrize("cov", [0.5, np.array([0.5]), np.array([[0.5]])])
def test_gaussian_mean_shape_1d(mean, cov):
    """Test that different types of one-dimensional means and covariances yield one-
    dimensional Gaussian measures."""
    measure = quad.GaussianMeasure(mean, cov)
    assert measure.dim == 1
