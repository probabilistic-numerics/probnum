"""Tests for random variable arithmetic."""

import numpy as np
import pytest

from probnum import randvars, utils


@pytest.fixture
def constant():
    return randvars.Constant(support=np.arange(10, 12))


@pytest.fixture
def normal(cov_cholesky):
    return randvars.Normal(
        mean=np.arange(4, 6), cov=np.diag(np.arange(5, 7)), cov_cholesky=cov_cholesky
    )


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_sum_left(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    sum_of_rvs = constant + normal

    np.testing.assert_allclose(sum_of_rvs.mean, constant.support + normal.mean)
    np.testing.assert_allclose(sum_of_rvs.cov, normal.cov)
    assert sum_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if sum_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(sum_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_sum_right(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    sum_of_rvs = normal + constant
    np.testing.assert_allclose(sum_of_rvs.mean, normal.mean + constant.support)
    np.testing.assert_allclose(sum_of_rvs.cov, normal.cov)
    assert sum_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if sum_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(sum_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_subtraction_left(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    diff_of_rvs = constant - normal

    np.testing.assert_allclose(diff_of_rvs.mean, constant.support - normal.mean)
    np.testing.assert_allclose(diff_of_rvs.cov, normal.cov)
    assert diff_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if diff_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(diff_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_subtraction_right(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    diff_of_rvs = normal - constant
    np.testing.assert_allclose(diff_of_rvs.mean, normal.mean - constant.support)
    np.testing.assert_allclose(diff_of_rvs.cov, normal.cov)
    assert diff_of_rvs.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    if diff_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(diff_of_rvs.cov_cholesky, normal.cov_cholesky)


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_matrix_multiplication_right(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    matrix_product = constant @ normal
    np.testing.assert_allclose(matrix_product.mean, constant.support @ normal.mean)
    np.testing.assert_allclose(
        matrix_product.cov, constant.support @ normal.cov @ constant.support.T
    )
    assert (
        matrix_product.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    )
    if matrix_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            matrix_product.cov_cholesky,
            utils.linalg.cholesky_update(constant.support @ normal.cov_cholesky),
        )


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_matrix_multiplication_left(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    matrix_product = normal @ constant
    np.testing.assert_allclose(matrix_product.mean, normal.mean @ constant.support)
    np.testing.assert_allclose(
        matrix_product.cov, constant.support.T @ normal.cov @ constant.support
    )
    assert (
        matrix_product.cov_cholesky_is_precomputed == normal.cov_cholesky_is_precomputed
    )
    if matrix_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            matrix_product.cov_cholesky,
            utils.linalg.cholesky_update(constant.support.T @ normal.cov_cholesky),
        )


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_multiplication_right(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""

    # Elementwise multiplication only supported for Constant RVs of size 1.
    scalar_like_constant = constant[0]
    element_product = scalar_like_constant * normal

    np.testing.assert_allclose(
        element_product.mean, scalar_like_constant.support * normal.mean
    )
    np.testing.assert_allclose(
        element_product.cov,
        scalar_like_constant.support ** 2 * normal.cov,
    )
    assert (
        element_product.cov_cholesky_is_precomputed
        == normal.cov_cholesky_is_precomputed
    )
    if element_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            element_product.cov_cholesky,
            scalar_like_constant.support * normal.cov_cholesky,
        )


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_multiplication_left(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""

    # Elementwise multiplication only supported for Constant RVs of size 1.
    scalar_like_constant = constant[0]
    element_product = normal * scalar_like_constant

    np.testing.assert_allclose(
        element_product.mean, normal.mean * scalar_like_constant.support
    )
    np.testing.assert_allclose(
        element_product.cov,
        normal.cov * scalar_like_constant.support ** 2,
    )
    assert (
        element_product.cov_cholesky_is_precomputed
        == normal.cov_cholesky_is_precomputed
    )
    if element_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            element_product.cov_cholesky,
            normal.cov_cholesky * scalar_like_constant.support,
        )


@pytest.mark.parametrize("cov_cholesky", [None, np.diag(np.sqrt(np.arange(5, 7)))])
def test_constant_normal_division(constant, normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""

    # Elementwise multiplication only supported for Constant RVs of size 1.
    scalar_like_constant = constant[0]
    element_product = normal / scalar_like_constant

    np.testing.assert_allclose(
        element_product.mean, normal.mean / scalar_like_constant.support
    )
    np.testing.assert_allclose(
        element_product.cov,
        normal.cov / scalar_like_constant.support ** 2,
    )
    assert (
        element_product.cov_cholesky_is_precomputed
        == normal.cov_cholesky_is_precomputed
    )
    if element_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            element_product.cov_cholesky,
            normal.cov_cholesky / scalar_like_constant.support,
        )
