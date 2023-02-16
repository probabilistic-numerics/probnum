"""Test cases for ``Kernel.matrix`` and ``Kernel.linop``"""

from typing import Callable, Optional

import numpy as np
import pytest

import probnum as pn
from probnum.typing import ShapeType


@pytest.fixture
def linop(
    kernel: pn.randprocs.kernels.Kernel, x0: np.ndarray, x1: Optional[np.ndarray]
) -> pn.linops.LinearOperator:
    """`LinearOperator` representation of the covariance matrix."""

    if x1 is None and np.prod(x0.shape[:-1]) >= 100:
        pytest.skip("Runs too long")

    return kernel.linop(x0, x1)


@pytest.fixture
def matrix(
    kernel: pn.randprocs.kernels.Kernel, x0: np.ndarray, x1: Optional[np.ndarray]
) -> np.ndarray:
    """Covariance matrix."""
    if x1 is None and np.prod(x0.shape[:-1]) >= 100:
        pytest.skip("Runs too long")

    return kernel.matrix(x0, x1)


@pytest.fixture
def matrix_naive(
    kernel: pn.randprocs.kernels.Kernel,
    kernel_call_naive: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    x0: np.ndarray,
    x1: Optional[np.ndarray],
) -> np.ndarray:
    """Reference covariance matrix"""
    if x1 is None:
        if np.prod(x0.shape[:-1]) >= 100:
            pytest.skip("Runs too long")

        x1 = x0

    return kernel_call_naive(
        x0=x0.reshape((-1, 1) + kernel.input_shape, order="C"),
        x1=x1.reshape((1, -1) + kernel.input_shape, order="C"),
    )


def test_linop_type(linop: pn.linops.LinearOperator):
    """Check whether a `Kernel.linop` evaluates to a `pn.linops.LinearOperator`."""

    assert isinstance(linop, pn.linops.LinearOperator)


def test_matrix_type(matrix: np.ndarray):
    """Check whether a `Kernel.matrix` evaluates to a numpy scalar or array."""

    assert isinstance(matrix, (np.ndarray, np.number))


def test_shape(
    kernel: pn.randprocs.kernels.Kernel,
    x0: np.ndarray,
    x1: Optional[np.ndarray],
    linop: pn.linops.LinearOperator,
    matrix: np.ndarray,
    matrix_naive: np.ndarray,
):
    """Test the shape of `Kernel.{linop,matrix}`."""

    assert linop.shape == matrix_naive.shape
    assert matrix.shape == matrix_naive.shape, (
        f"Kernel {type(kernel)} does not have the right shape if evaluated at inputs "
        f"with x0.shape={x0.shape}"
        + ("" if x1 is None else f"and x1.shape={x1.shape}.")
    )


def test_linop_equals_matrix_naive(
    linop: pn.linops.LinearOperator,
    matrix_naive: np.ndarray,
):
    """Test whether the values of `Kernel.linop(...).todense()` match the reference
    implementation."""

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
    """Test whether the values of `Kernel.matrix` match the reference implementation."""

    np.testing.assert_array_equal(
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
def test_input_shape_mismatch(kernel: pn.randprocs.kernels.Kernel, shape: ShapeType):
    """Test whether passing an input with the wrong input shape raises an error."""

    if kernel.input_ndim > 0:
        input_shape = shape + tuple(dim + 1 for dim in kernel.input_shape)

        # `Kernel.linop`
        with pytest.raises(ValueError):
            kernel.linop(np.zeros(input_shape))

        with pytest.raises(ValueError):
            kernel.linop(np.ones(input_shape), np.zeros(shape + kernel.input_shape))

        with pytest.raises(ValueError):
            kernel.linop(np.ones(shape + kernel.input_shape), np.zeros(input_shape))

        # `Kernel.matrix`
        with pytest.raises(ValueError):
            kernel.matrix(np.zeros(input_shape))

        with pytest.raises(ValueError):
            kernel.matrix(np.ones(input_shape), np.zeros(shape + kernel.input_shape))

        with pytest.raises(ValueError):
            kernel.matrix(np.ones(shape + kernel.input_shape), np.zeros(input_shape))
