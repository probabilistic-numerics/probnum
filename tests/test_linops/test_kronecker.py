"""Tests for Kronecker-type linear operators."""

import numpy as np
import pytest
import pytest_cases

import probnum as pn


@pytest_cases.parametrize_with_cases(
    "linop,matrix",
    cases=".test_linops_cases.kronecker_cases",
    has_tag="symmetric_kronecker",
)
def test_symmetric_kronecker_commutative(
    linop: pn.linops.SymmetricKronecker, matrix: np.ndarray
):
    linop_commuted = pn.linops.SymmetricKronecker(linop.B, linop.A)

    np.testing.assert_array_equal(linop.todense(), linop_commuted.todense())
    np.testing.assert_almost_equal(linop_commuted.todense(), matrix)


@pytest.mark.parametrize(
    "A,B", [(np.array([[5, 1], [1, 10]]), np.array([[-2, 0.1], [0.1, 8]]))]
)
def test_symmetric_kronecker_symmetric_factors(A, B):
    """Dense matrix from symmetric Kronecker product of two symmetric matrices must be
    symmetric."""
    linop = pn.linops.SymmetricKronecker(A, B)
    linop_transpose = linop.T
    linop_dense = linop.todense()

    np.testing.assert_array_equal(linop_dense, linop_dense.T)
    np.testing.assert_array_equal(linop_dense, linop_transpose.todense())


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
def test_symmetrize(n):

    rng = np.random.default_rng(42)
    x = rng.uniform(size=n * n)
    X = np.reshape(x, (n, n))
    y = pn.linops.Symmetrize(n) @ x

    np.testing.assert_array_equal(
        y.reshape(n, n), 0.5 * (X + X.T), err_msg="Matrix not symmetric."
    )

    Z = rng.uniform(size=(9, 5))
    W = pn.linops.Symmetrize(3) @ Z

    np.testing.assert_array_equal(
        W,
        np.vstack([pn.linops.Symmetrize(3) @ col for col in Z.T]).T,
        err_msg="Matrix columns were not symmetrized.",
    )

    np.testing.assert_array_equal(
        np.shape(W),
        np.shape(Z),
        err_msg="Symmetrized matrix columns do not have the right shape.",
    )
