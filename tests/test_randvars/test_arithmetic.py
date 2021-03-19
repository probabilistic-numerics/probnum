"""Tests for random variable arithmetic."""

import numpy as np
import pytest

from probnum import randvars


@pytest.fixture
def constant():
    return randvars.Constant(support=np.arange(10, 12))


@pytest.fixture
def normal(cov_cholesky):
    return randvars.Normal(
        mean=np.arange(2), cov=np.diag(np.arange(5, 7)), cov_cholesky=cov_cholesky
    )


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_sum_left(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    sum_of_rvs = constant + normal

    np.testing.assert_allclose(sum_of_rvs.mean, constant.mean + normal.mean)
    np.testing.assert_allclose(sum_of_rvs.cov, normal.cov)
    assert sum_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if sum_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(sum_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_sum_right(constant, normal):
    sum_of_rvs = normal + constant
    np.testing.assert_allclose(sum_of_rvs.mean, normal.mean + constant.mean)
    np.testing.assert_allclose(sum_of_rvs.cov, normal.cov)
    assert sum_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if sum_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(sum_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_subtraction_left(constant, normal):
    diff_of_rvs = constant - normal

    np.testing.assert_allclose(diff_of_rvs.mean, constant.mean - normal.mean)
    np.testing.assert_allclose(diff_of_rvs.cov, normal.cov)
    assert diff_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if diff_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(diff_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_subtraction_right(constant, normal):
    diff_of_rvs = normal - constant
    np.testing.assert_allclose(diff_of_rvs.mean, normal.mean - constant.mean)
    np.testing.assert_allclose(diff_of_rvs.cov, normal.cov)
    assert diff_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if diff_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(diff_of_rvs.cov_cholesky, normal.cov_cholesky)
