"""Tests for matrix-variate normal arithmetic."""

import numpy as np
import pytest

from probnum import utils


@pytest.mark.parametrize("shape_const,shape", [((4, 3), (3, 2)), ((1, 3), (3, 2))])
@pytest.mark.parametrize("precompute_cov_cholesky", [False])
def test_constant_matrixvariate_normal_matrix_multiplication_right(
    constant, matrixvariate_normal
):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    matrix_product = constant @ matrixvariate_normal
    np.testing.assert_allclose(
        matrix_product.mean, constant.support @ matrixvariate_normal.mean
    )
    kron_prod = np.kron(constant.support, np.identity(matrixvariate_normal.shape[1]))
    np.testing.assert_allclose(
        matrix_product.cov.todense(),
        kron_prod @ matrixvariate_normal.cov @ kron_prod.T,
    )


@pytest.mark.parametrize(
    "shape,shape_const", [((4, 3), (3, 2)), ((2, 2), (2,)), ((3, 2), (2, 1))]
)
@pytest.mark.parametrize("precompute_cov_cholesky", [False])
def test_constant_matrixvariate_normal_matrix_multiplication_left(
    constant, matrixvariate_normal
):
    """Assert that mean and covariance follow the correct formula and that a Cholesky
    factor is preserved if it existed before."""
    matrix_product = matrixvariate_normal @ constant
    np.testing.assert_allclose(
        matrix_product.mean, matrixvariate_normal.mean @ constant.support
    )
    if constant.support.ndim == 1:
        constant_support = constant.support[:, None]
    else:
        constant_support = constant.support
    kron_prod = np.kron(np.identity(matrixvariate_normal.shape[0]), constant_support)
    np.testing.assert_allclose(
        matrix_product.cov.todense(),
        kron_prod.T @ matrixvariate_normal.cov @ kron_prod,
    )
