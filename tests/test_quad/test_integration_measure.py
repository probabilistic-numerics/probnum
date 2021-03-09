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
