"""Test cases for integration measures."""

import numpy as np
import pytest

from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure


# Tests for Gaussian measure
def test_gaussian_diagonal_covariance(input_dim: int):
    """Check that diagonal covariance matrices are recognised as diagonal."""
    mean = np.full((input_dim,), 0.0)
    cov = np.eye(input_dim)
    measure = GaussianMeasure(mean, cov)
    assert measure.diagonal_covariance


@pytest.mark.parametrize("input_dim_non_diagonal", [2, 10, 100])
def test_gaussian_non_diagonal_covariance(input_dim_non_diagonal):
    """Check that non-diagonal covariance matrices are recognised as non-diagonal."""
    mean = np.full((input_dim_non_diagonal,), 0.0)
    cov = np.eye(input_dim_non_diagonal)
    cov[0, 1] = 1.5
    measure = GaussianMeasure(mean, cov)
    assert not measure.diagonal_covariance


@pytest.mark.parametrize("mean", [0, np.array([0]), np.array([[0]])])
@pytest.mark.parametrize("cov", [0.5, np.array([0.5]), np.array([[0.5]])])
def test_gaussian_mean_shape_1d(mean, cov):
    """Test that different types of one-dimensional means and covariances yield one-
    dimensional Gaussian measures when no dimension is given."""
    measure = GaussianMeasure(mean=mean, cov=cov)
    assert measure.input_dim == 1
    assert measure.mean.size == 1
    assert measure.cov.size == 1


@pytest.mark.parametrize("neg_dim", [0, -1, -10, -100])
def test_gaussian_negative_dimension(neg_dim):
    """Make sure that a negative dimension raises ValueError."""
    with pytest.raises(ValueError):
        GaussianMeasure(0, 1, neg_dim)


def test_gaussian_param_assignment(input_dim: int):
    """Check that diagonal mean and covariance for higher dimensions are extended
    correctly."""
    measure = GaussianMeasure(0, 1, input_dim)
    if input_dim == 1:
        assert measure.mean == 0.0
        assert measure.cov == 1.0
        assert isinstance(measure.mean, np.ndarray)
        assert isinstance(measure.cov, np.ndarray)
    else:
        assert np.array_equal(measure.mean, np.zeros(input_dim))
        assert np.array_equal(measure.cov, np.eye(input_dim))


def test_gaussian_scalar():
    """Check that the 1d Gaussian case works."""
    measure = GaussianMeasure(0.5, 1.5)
    assert measure.mean == 0.5
    assert measure.cov == 1.5


# Tests for Lebesgue measure
def test_lebesgue_dim_correct(input_dim: int):
    """Check that dimensions are handled correctly."""
    domain1 = (0.0, 1.87)
    measure11 = LebesgueMeasure(domain=domain1)
    measure12 = LebesgueMeasure(input_dim=input_dim, domain=domain1)

    assert measure11.input_dim == 1
    assert measure12.input_dim == input_dim

    domain2 = (np.full((input_dim,), -0.1), np.full((input_dim,), 0.0))
    measure21 = LebesgueMeasure(domain=domain2)
    measure22 = LebesgueMeasure(input_dim=input_dim, domain=domain2)

    assert measure21.input_dim == input_dim
    assert measure22.input_dim == input_dim


@pytest.mark.parametrize("domain_a", [0, np.full((3,), 0), np.full((13,), 0)])
@pytest.mark.parametrize("domain_b", [np.full((4,), 1.2), np.full((14,), 1.2)])
@pytest.mark.parametrize("input_dim", [-10, -2, 0, 2, 12, 122])
def test_lebesgue_dim_incorrect(domain_a, domain_b, input_dim):
    """Check that ValueError is raised if domain limits have mismatching dimensions or
    dimension is not positive."""
    with pytest.raises(ValueError):
        LebesgueMeasure(domain=(domain_a, domain_b), input_dim=input_dim)


def test_lebesgue_normalization(input_dim: int):
    """Check that normalization constants are handled properly when not equal to one."""
    domain = (0, 2)
    measure = LebesgueMeasure(domain=domain, input_dim=input_dim, normalized=True)

    volume = 2**input_dim
    assert measure.normalization_constant == 1 / volume


@pytest.mark.parametrize("domain", [(0, np.Inf), (-np.Inf, 0), (-np.Inf, np.Inf)])
def test_lebesgue_normalization_raises(domain, input_dim: int):
    """Check that exception is raised when normalization is not possible."""
    with pytest.raises(ValueError):
        LebesgueMeasure(domain=domain, input_dim=input_dim, normalized=True)


def test_lebesgue_unnormalized(input_dim: int):
    """Check that normalization constants are handled properly when equal to one."""
    measure1 = LebesgueMeasure(domain=(0, 1), input_dim=input_dim, normalized=True)
    measure2 = LebesgueMeasure(domain=(0, 1), input_dim=input_dim, normalized=False)
    assert measure1.normalization_constant == measure2.normalization_constant


# Tests for all integration measures
def test_density_call(x, measure):
    expected_shape = (x.shape[0],)
    assert measure(x).shape == expected_shape
