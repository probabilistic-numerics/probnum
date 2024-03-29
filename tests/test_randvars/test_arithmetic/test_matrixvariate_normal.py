"""Tests for matrix-variate normal arithmetic."""

import numpy as np
import pytest

from probnum import linops


@pytest.mark.parametrize(
    "shape_const,shape",
    [
        ((4, 3), (3, 2)),
        ((2,), (2, 3)),
        ((3, 2), (2, 1)),
        ((1, 2), (2, 1)),
        ((2, 1), (1, 2)),
        ((1, 1), (1, 1)),
    ],
)
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

    matrix_product_cov = matrix_product.cov
    if isinstance(matrix_product_cov, linops.LinearOperator):
        matrix_product_cov = matrix_product_cov.todense()
    kron_prod = np.kron(constant.support, np.identity(matrixvariate_normal.shape[1]))
    np.testing.assert_allclose(
        matrix_product_cov,
        kron_prod @ matrixvariate_normal.cov @ kron_prod.T,
    )


@pytest.mark.parametrize(
    "shape,shape_const",
    [
        ((4, 3), (3, 2)),
        ((3, 2), (2,)),
        ((3, 2), (2, 1)),
        ((1, 2), (2, 1)),
        ((2, 1), (1, 2)),
        ((1, 1), (1,)),
    ],
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

    matrix_product_cov = matrix_product.cov
    if isinstance(matrix_product_cov, linops.LinearOperator):
        matrix_product_cov = matrix_product_cov.todense()
    kron_prod = np.kron(np.identity(matrixvariate_normal.shape[0]), constant_support.T)

    np.testing.assert_allclose(
        matrix_product_cov,
        kron_prod @ matrixvariate_normal.cov @ kron_prod.T,
    )
