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
