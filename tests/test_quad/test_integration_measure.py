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
    dimensional Gaussian measures when no dimension is given."""
    measure = quad.GaussianMeasure(mean, cov)
    assert measure.dim == 1


@pytest.mark.parametrize("neg_dim", [0, -1, -10, -100])
def test_gaussian_negative_dimension(neg_dim):
    """Make sure that a negative dimension raises ValueError."""
    with pytest.raises(ValueError):
        measure = quad.GaussianMeasure(0, 1, neg_dim)


@pytest.mark.parametrize("input_dim_for_scalar", [2, 10, 100])
def test_gaussian_scalar_nd(input_dim_for_scalar: int):
    """Check that diagonal mean and covariance for higher dimensions are extended
    correctly."""
    measure = quad.GaussianMeasure(0, 1, input_dim_for_scalar)
    assert np.array_equal(measure.mean, np.zeros(input_dim_for_scalar))
    assert np.array_equal(measure.cov, np.eye(input_dim_for_scalar))


@pytest.mark.parametrize(
    "cov_vector", [np.array([1, 1]), np.array([0.1, 1, 9.8]), np.full((98,), 0.2)]
)
def test_gaussian_vector_cov(cov_vector):
    dim = cov_vector.size
    mean = np.zeros(dim)
    measure = quad.GaussianMeasure(mean, cov_vector)
    assert measure.cov.shape == (dim, dim)


def test_gaussian_scalar():
    """Check that the 1d case works."""
    measure = quad.GaussianMeasure(0.5, 1.5)
    assert measure.mean == 0.5
    assert measure.cov == 1.5
