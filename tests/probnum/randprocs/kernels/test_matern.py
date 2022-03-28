"""Test cases for the Matern kernel."""

import pytest

from probnum import backend, compat
from probnum.backend.typing import ShapeType
from probnum.randprocs import kernels


@pytest.mark.parametrize("nu", [-1, -1.0, 0.0, 0])
def test_nonpositive_nu_raises_exception(nu):
    """Check whether a non-positive nu parameter raises a ValueError."""
    with pytest.raises(ValueError):
        kernels.Matern(input_shape=(), nu=nu)


def test_nu_large_recovers_rbf_kernel(
    x0: backend.Array, x1: backend.Array, input_shape: ShapeType
):
    """Test whether a Matern kernel with nu large is close to an RBF kernel."""
    lengthscale = 1.25
    rbf = kernels.ExpQuad(input_shape=input_shape, lengthscale=lengthscale)
    matern = kernels.Matern(input_shape=input_shape, lengthscale=lengthscale, nu=15)

    compat.testing.assert_allclose(
        rbf.matrix(x0, x1),
        matern.matrix(x0, x1),
        err_msg="RBF and Matern kernel are not sufficiently close for nu->infty.",
        rtol=0.05,
        atol=0.01,
    )
