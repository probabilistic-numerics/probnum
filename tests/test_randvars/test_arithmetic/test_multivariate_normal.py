"""Tests for multi-variate normal arithmetic."""

import numpy as np
import pytest

from probnum import utils


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_sum_left(constant, multivariate_normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    sum_of_rvs = constant + multivariate_normal

    np.testing.assert_allclose(
        sum_of_rvs.mean, constant.support + multivariate_normal.mean
    )
    np.testing.assert_allclose(sum_of_rvs.cov, multivariate_normal.cov)
    assert (
        sum_of_rvs.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if sum_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            sum_of_rvs.cov_cholesky, multivariate_normal.cov_cholesky
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_sum_right(constant, multivariate_normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    sum_of_rvs = multivariate_normal + constant
    np.testing.assert_allclose(
        sum_of_rvs.mean, multivariate_normal.mean + constant.support
    )
    np.testing.assert_allclose(sum_of_rvs.cov, multivariate_normal.cov)
    assert (
        sum_of_rvs.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if sum_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            sum_of_rvs.cov_cholesky, multivariate_normal.cov_cholesky
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_subtraction_left(constant, multivariate_normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    diff_of_rvs = constant - multivariate_normal

    np.testing.assert_allclose(
        diff_of_rvs.mean, constant.support - multivariate_normal.mean
    )
    np.testing.assert_allclose(diff_of_rvs.cov, multivariate_normal.cov)
    assert (
        diff_of_rvs.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if diff_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            diff_of_rvs.cov_cholesky, multivariate_normal.cov_cholesky
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_subtraction_right(constant, multivariate_normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    diff_of_rvs = multivariate_normal - constant
    np.testing.assert_allclose(
        diff_of_rvs.mean, multivariate_normal.mean - constant.support
    )
    np.testing.assert_allclose(diff_of_rvs.cov, multivariate_normal.cov)
    assert (
        diff_of_rvs.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if diff_of_rvs.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            diff_of_rvs.cov_cholesky, multivariate_normal.cov_cholesky
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_matrix_multiplication_right(
    constant, multivariate_normal
):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    matrix_product = constant @ multivariate_normal
    np.testing.assert_allclose(
        matrix_product.mean, constant.support @ multivariate_normal.mean
    )
    np.testing.assert_allclose(
        matrix_product.cov,
        constant.support @ multivariate_normal.cov @ constant.support.T,
    )
    assert (
        matrix_product.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if matrix_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            matrix_product.cov_cholesky,
            utils.linalg.cholesky_update(
                constant.support @ multivariate_normal.cov_cholesky
            ),
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_matrix_multiplication_left(
    constant, multivariate_normal
):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    matrix_product = multivariate_normal @ constant
    np.testing.assert_allclose(
        matrix_product.mean, multivariate_normal.mean @ constant.support
    )
    np.testing.assert_allclose(
        matrix_product.cov,
        constant.support.T @ multivariate_normal.cov @ constant.support,
    )
    assert (
        matrix_product.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if matrix_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            matrix_product.cov_cholesky,
            utils.linalg.cholesky_update(
                constant.support.T @ multivariate_normal.cov_cholesky
            ),
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_multiplication_right(
    constant, multivariate_normal
):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""

    # Elementwise multiplication only supported for Constant RVs of size 1.
    scalar_like_constant = constant[0]
    element_product = scalar_like_constant * multivariate_normal

    np.testing.assert_allclose(
        element_product.mean, scalar_like_constant.support * multivariate_normal.mean
    )
    np.testing.assert_allclose(
        element_product.cov,
        scalar_like_constant.support ** 2 * multivariate_normal.cov,
    )
    assert (
        element_product.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if element_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            element_product.cov_cholesky,
            scalar_like_constant.support * multivariate_normal.cov_cholesky,
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_multiplication_left(
    constant, multivariate_normal
):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""

    # Elementwise multiplication only supported for Constant RVs of size 1.
    scalar_like_constant = constant[0]
    element_product = multivariate_normal * scalar_like_constant

    np.testing.assert_allclose(
        element_product.mean, multivariate_normal.mean * scalar_like_constant.support
    )
    np.testing.assert_allclose(
        element_product.cov,
        multivariate_normal.cov * scalar_like_constant.support ** 2,
    )
    assert (
        element_product.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if element_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            element_product.cov_cholesky,
            multivariate_normal.cov_cholesky * scalar_like_constant.support,
        )


@pytest.mark.parametrize("shape,shape_const", [((3,), (3,))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False, True])
def test_constant_multivariate_normal_division(constant, multivariate_normal):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""

    # Elementwise multiplication only supported for Constant RVs of size 1.
    scalar_like_constant = constant[0]
    element_product = multivariate_normal / scalar_like_constant

    np.testing.assert_allclose(
        element_product.mean, multivariate_normal.mean / scalar_like_constant.support
    )
    np.testing.assert_allclose(
        element_product.cov,
        multivariate_normal.cov / scalar_like_constant.support ** 2,
    )
    assert (
        element_product.cov_cholesky_is_precomputed
        == multivariate_normal.cov_cholesky_is_precomputed
    )
    if element_product.cov_cholesky_is_precomputed:
        np.testing.assert_allclose(
            element_product.cov_cholesky,
            multivariate_normal.cov_cholesky / scalar_like_constant.support,
        )
