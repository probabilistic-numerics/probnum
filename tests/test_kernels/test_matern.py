"""Test cases for the Matern kernel."""

import numpy as np
import pytest

from probnum import kernels


@pytest.mark.parametrize("nu", [-1, -1.0, 0.0, 0])
def test_nonpositive_nu_raises_exception(nu):
    """Check whether a non-positive nu parameter raises a ValueError."""
    with pytest.raises(ValueError):
        kernels.Matern(input_dim=1, nu=nu)


def test_nu_large_recovers_rbf_kernel(x0: np.ndarray, x1: np.ndarray, input_dim: int):
    """Test whether a Matern kernel with nu large is close to an RBF kernel."""
    lengthscale = 1.25
    rbf = kernels.ExpQuad(lengthscale=lengthscale, input_dim=input_dim)
    matern = kernels.Matern(lengthscale=lengthscale, nu=15, input_dim=input_dim)

    if x1 is None:
        kernmat_rbf = rbf(x0)
        kernmat_matern = matern(x0)
    else:
        kernmat_rbf = rbf(x0[:, None, :], x1[None, :, :])
        kernmat_matern = matern(x0[:, None, :], x1[None, :, :])

    np.testing.assert_allclose(
        kernmat_matern,
        kernmat_rbf,
        err_msg="RBF and Matern kernel are not equivalent for nu=infty.",
        rtol=0.05,
        atol=0.01,
    )
