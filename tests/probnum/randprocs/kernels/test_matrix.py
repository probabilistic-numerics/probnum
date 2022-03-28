"""Test cases for ``Kernel.matrix``"""

from typing import Callable, Optional

import pytest

from probnum import backend, compat
from probnum.backend.typing import ShapeType
from probnum.randprocs import kernels


@pytest.fixture(name="kernmat", scope="module")
def fixture_kernmat(
    kernel: kernels.Kernel, x0: backend.Array, x1: Optional[backend.Array]
) -> backend.Array:
    """Kernel evaluated at the data."""
    if x1 is None and x0.size // kernel.input_size >= 100:
        pytest.skip("Runs too long")

    return kernel.matrix(x0, x1)


@pytest.fixture(name="kernmat_naive", scope="module")
def fixture_kernmat_naive(
    kernel: kernels.Kernel,
    kernel_call_naive: Callable[
        [backend.Array, Optional[backend.Array]], backend.Array
    ],
    x0: backend.Array,
    x1: Optional[backend.Array],
) -> backend.Array:
    """Kernel evaluated at the data."""

    if x1 is None:
        if x0.size // kernel.input_size >= 100:
            pytest.skip("Runs too long")

        x1 = x0

    if x0.ndim > kernel.input_ndim and x1.ndim > kernel.input_ndim:
        return kernel_call_naive(x0=x0[:, None], x1=x1[None, :])

    return kernel_call_naive(x0, x1)


def test_type(kernmat: backend.Array):
    """Check whether a kernel evaluates to a numpy scalar or array."""

    assert backend.isarray(kernmat)


def test_shape(
    kernel: kernels.Kernel,
    x0: backend.Array,
    x1: Optional[backend.Array],
    kernmat: backend.Array,
    kernmat_naive: backend.Array,
):
    """Test the shape of a kernel evaluated at sets of inputs."""

    assert kernmat.shape == kernmat_naive.shape, (
        f"Kernel {type(kernel)} does not have the right shape if evaluated at inputs "
        f"with x0.shape={x0.shape}"
        + ("" if x1 is None else f"and x1.shape={x1.shape}.")
    )


def test_kernel_matrix_against_naive(
    kernmat: backend.Array,
    kernmat_naive: backend.Array,
):
    """Test the computation of the kernel matrix against a naive computation."""

    compat.testing.assert_allclose(
        kernmat,
        kernmat_naive,
        rtol=10**-12,
        atol=10**-12,
    )


@pytest.mark.parametrize(
    "x0_shape,x1_shape",
    [
        ((2, 5), (3, 5)),
        ((4, 4), (4, 2)),
    ],
)
def test_invalid_shape(
    kernel: kernels.Kernel,
    x0_shape: backend.Array,
    x1_shape: backend.Array,
):
    """Test whether an error is raised if the inputs can not be broadcast to a common
    shape."""

    with pytest.raises(ValueError):
        kernel.matrix(backend.zeros(x0_shape + kernel.input_shape))

    with pytest.raises(ValueError):
        kernel.matrix(
            backend.zeros(x0_shape + kernel.input_shape),
            backend.ones(x1_shape + kernel.input_shape),
        )


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (10,),
    ],
)
def test_wrong_input_dimension(kernel: kernels.Kernel, shape: ShapeType):
    """Test whether passing an input with the wrong input dimension raises an error."""

    if kernel.input_ndim == 0:
        input_shape = shape + (4, 2)
    else:
        input_shape = shape + tuple(dim + 1 for dim in kernel.input_shape)

    with pytest.raises(ValueError):
        kernel.matrix(backend.zeros(input_shape))

    with pytest.raises(ValueError):
        kernel.matrix(
            backend.ones(input_shape), backend.zeros(shape + kernel.input_shape)
        )

    with pytest.raises(ValueError):
        kernel.matrix(
            backend.ones(shape + kernel.input_shape), backend.zeros(input_shape)
        )
