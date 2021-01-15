"""Test cases for kernels."""

import numpy as np
import pytest
import scipy.spatial

import probnum.kernels as kernels
import probnum.utils as _utils


def test_1d_data(kernel: kernels.Kernel, x0_1d: np.ndarray, input_dim: int):
    """Test handling of 1d data."""
    kernmat = kernel(x0_1d)
    assert isinstance(kernmat, (float, np.float_))


def test_shape(
    kernel: kernels.Kernel, kernmat: np.ndarray, x0: np.ndarray, x1: np.ndarray
):
    """Test the shape of a kernel evaluated at sets of inputs."""

    # Check shape
    if x1 is None:
        x1 = x0
    if (x0.ndim == 0 and x1.ndim == 0) or (x0.ndim == 1 and x1.ndim == 1):
        kern_shape = ()
    else:
        kern_shape = (x0.shape[0], x1.shape[0])
    if kernel.output_dim > 1:
        kern_shape += (kernel.output_dim, kernel.output_dim)

    assert kernmat.shape == kern_shape, (
        f"Kernel {type(kernel)} does not have the "
        f"right shape if evaluated at inputs of x0.shape={x0.shape} and x1.shape={x1.shape}."
    )


def test_type(kernmat: np.ndarray):
    """Check whether a kernel evaluates to a numpy scalar or array."""
    assert isinstance(kernmat, np.ndarray) or np.isscalar(kernmat)


def test_kernel_matrix_against_naive(
    kernel: kernels.Kernel, kernmat: np.ndarray, x0: np.ndarray, x1: np.ndarray
):
    """Test the computation of the kernel matrix against a naive computation."""
    if x1 is None:
        x1 = x0
    np.testing.assert_allclose(
        kernmat,
        scipy.spatial.distance.cdist(
            x0,
            x1,
            metric=lambda x0, x1, k=kernel: _utils.as_numpy_scalar(k(x0, x1).item()),
        ),
        rtol=10 ** -12,
        atol=10 ** -12,
    )


@pytest.mark.parametrize("input_dim", [2], indirect=True)
@pytest.mark.parametrize(
    "x0,x1",
    [
        (1, 1),
        (1.0, np.array([1.0, 0.0])),
        (np.array([1.0, 0.0, 0.2]), np.array([1.0, 0.0, 2.3])),
        (np.array([[1.0]]), np.array([[1.0, -1.0]])),
    ],
)
def test_misshaped_input(
    kernel: kernels.Kernel, input_dim: int, x0: np.ndarray, x1: np.ndarray
):
    """Test whether misshaped/mismatched input raises an error."""
    with pytest.raises(ValueError):
        kernel(x0, x1)
    with pytest.raises(ValueError):
        kernel(x0)
