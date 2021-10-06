"""Test cases for ``Kernel.matrix``"""

from typing import Callable, Optional

import numpy as np
import pytest

import probnum as pn
from probnum.typing import ShapeType


@pytest.fixture(name="kernmat")
def fixture_kernmat(
    kernel: pn.kernels.Kernel, x0: np.ndarray, x1: Optional[np.ndarray]
) -> np.ndarray:
    """Kernel evaluated at the data."""
    if x1 is None and np.prod(x0.shape[:-1]) >= 100:
        pytest.skip("Runs too long")

    return kernel.matrix(x0, x1)


@pytest.fixture(name="kernmat_naive")
def fixture_kernmat_naive(
    kernel_call_naive: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    x0: np.ndarray,
    x1: Optional[np.ndarray],
) -> np.ndarray:
    """Kernel evaluated at the data."""

    if x1 is None:
        if np.prod(x0.shape[:-1]) >= 100:
            pytest.skip("Runs too long")

        return kernel_call_naive(x0=x0[:, None, :], x1=x0[None, :, :])

    return kernel_call_naive(x0=x0[:, None, :], x1=x1[None, :, :])


def test_type(kernmat: np.ndarray):
    """Check whether a kernel evaluates to a numpy scalar or array."""

    assert isinstance(kernmat, (np.ndarray, np.number))


def test_shape(
    kernel: pn.kernels.Kernel,
    x0: np.ndarray,
    x1: Optional[np.ndarray],
    kernmat: np.ndarray,
    kernmat_naive: np.ndarray,
):
    """Test the shape of a kernel evaluated at sets of inputs."""

    assert kernmat.shape == kernmat_naive.shape, (
        f"Kernel {type(kernel)} does not have the right shape if evaluated at inputs "
        f"with x0.shape={x0.shape}"
        + ("" if x1 is None else f"and x1.shape={x1.shape}.")
    )


def test_kernel_matrix_against_naive(
    kernmat: np.ndarray,
    kernmat_naive: np.ndarray,
):
    """Test the computation of the kernel matrix against a naive computation."""

    np.testing.assert_allclose(
        kernmat,
        kernmat_naive,
        rtol=10 ** -12,
        atol=10 ** -12,
    )


@pytest.mark.parametrize(
    "x0_shape,x1_shape",
    [
        ((2, 5), (3, 5)),
        ((4, 4), (4, 2)),
    ],
)
def test_invalid_shape(
    kernel: pn.kernels.Kernel,
    x0_shape: np.ndarray,
    x1_shape: np.ndarray,
):
    """Test whether an error is raised if the inputs can not be broadcast to a common
    shape."""

    with pytest.raises(ValueError):
        kernel.matrix(np.zeros(x0_shape + (kernel.input_dim,)))

    with pytest.raises(ValueError):
        kernel.matrix(
            np.zeros(x0_shape + (kernel.input_dim,)),
            np.ones(x1_shape + (kernel.input_dim,)),
        )


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (10,),
    ],
)
def test_wrong_input_dimension(kernel: pn.kernels.Kernel, shape: ShapeType):
    """Test whether passing an input with the wrong input dimension raises an error."""

    input_shape = shape + (kernel.input_dim + 1,)

    with pytest.raises(ValueError):
        kernel.matrix(np.zeros(input_shape))

    with pytest.raises(ValueError):
        kernel.matrix(np.ones(input_shape), np.zeros(shape + (kernel.input_dim,)))

    with pytest.raises(ValueError):
        kernel.matrix(np.ones(shape + (kernel.input_dim,)), np.zeros(input_shape))
