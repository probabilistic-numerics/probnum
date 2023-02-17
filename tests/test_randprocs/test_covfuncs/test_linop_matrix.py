"""Test cases for ``CovarianceFunction.matrix`` and ``CovarianceFunction.linop``"""

from typing import Callable, Optional

import numpy as np
import pytest

import probnum as pn
from probnum.typing import ShapeType


@pytest.fixture
def linop(
    k: pn.randprocs.covfuncs.CovarianceFunction,
    x0: np.ndarray,
    x1: Optional[np.ndarray],
) -> pn.linops.LinearOperator:
    """`LinearOperator` representation of the covariance matrix."""

    if x1 is None and np.prod(x0.shape[:-1]) >= 100:
        pytest.skip("Runs too long")

    return k.linop(x0, x1)


@pytest.fixture
def matrix(
    k: pn.randprocs.covfuncs.CovarianceFunction,
    x0: np.ndarray,
    x1: Optional[np.ndarray],
) -> np.ndarray:
    """Covariance matrix."""
    if x1 is None and np.prod(x0.shape[:-1]) >= 100:
        pytest.skip("Runs too long")

    return k.matrix(x0, x1)


@pytest.fixture
def matrix_naive(
    k: pn.randprocs.covfuncs.CovarianceFunction,
    k_call_naive: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    x0: np.ndarray,
    x1: Optional[np.ndarray],
) -> np.ndarray:
    """Reference covariance matrix"""
    if x1 is None:
        if np.prod(x0.shape[:-1]) >= 100:
            pytest.skip("Runs too long")

        x1 = x0

    return k_call_naive(
        x0=x0.reshape((-1, 1) + k.input_shape, order="C"),
        x1=x1.reshape((1, -1) + k.input_shape, order="C"),
    )


def test_linop_type(linop: pn.linops.LinearOperator):
    """Check whether a `CovarianceFunction.linop` evaluates to a
    `pn.linops.LinearOperator`."""

    assert isinstance(linop, pn.linops.LinearOperator)


def test_matrix_type(matrix: np.ndarray):
    """Check whether a `CovarianceFunction.matrix` evaluates to a numpy scalar or
    array."""

    assert isinstance(matrix, (np.ndarray, np.number))


def test_shape(
    k: pn.randprocs.covfuncs.CovarianceFunction,
    x0: np.ndarray,
    x1: Optional[np.ndarray],
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    matrix_naive: np.ndarray,
):
    """Test the shape of `CovarianceFunction.{linop,matrix}`."""

    assert linop.shape == matrix_naive.shape
    assert matrix.shape == matrix_naive.shape, (
        f"Covariance function {type(k)} does not have the right shape if evaluated at "
        f"inputs with x0.shape={x0.shape}"
        + ("" if x1 is None else f"and x1.shape={x1.shape}.")
    )


def test_linop_equals_matrix_naive(
    linop: pn.linops.LinearOperator,
    matrix_naive: np.ndarray,
):
    """Test whether the values of `CovarianceFunction.linop(...).todense()` match the
    reference implementation."""

    np.testing.assert_allclose(
        linop.todense(),
        matrix_naive,
        rtol=10**-12,
        atol=10**-12,
    )


def test_matrix_equals_matrix_naive(
    matrix: np.ndarray,
    matrix_naive: np.ndarray,
):
    """Test whether the values of `CovarianceFunction.matrix` match the reference
    implementation."""

    np.testing.assert_allclose(
        matrix,
        matrix_naive,
        rtol=10**-12,
        atol=10**-12,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (10,),
    ],
)
def test_input_shape_mismatch(
    k: pn.randprocs.covfuncs.CovarianceFunction, shape: ShapeType
):
    """Test whether passing an input with the wrong input shape raises an error."""

    if k.input_ndim > 0:
        input_shape = shape + tuple(dim + 1 for dim in k.input_shape)

        # `CovarianceFunction.linop`
        with pytest.raises(ValueError):
            k.linop(np.zeros(input_shape))

        with pytest.raises(ValueError):
            k.linop(np.ones(input_shape), np.zeros(shape + k.input_shape))

        with pytest.raises(ValueError):
            k.linop(np.ones(shape + k.input_shape), np.zeros(input_shape))

        # `CovarianceFunction.matrix`
        with pytest.raises(ValueError):
            k.matrix(np.zeros(input_shape))

        with pytest.raises(ValueError):
            k.matrix(np.ones(input_shape), np.zeros(shape + k.input_shape))

        with pytest.raises(ValueError):
            k.matrix(np.ones(shape + k.input_shape), np.zeros(input_shape))
