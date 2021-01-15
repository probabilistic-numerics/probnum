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
    kernmat_rbf = kernels.ExpQuad(lengthscale=lengthscale, input_dim=input_dim)
    kernmat_matern = kernels.Matern(lengthscale=lengthscale, nu=15, input_dim=input_dim)
    np.testing.assert_allclose(
        kernmat_rbf(x0, x1),
        kernmat_matern(x0, x1),
        err_msg="RBF and Matern kernel are not equivalent for nu=infty.",
        rtol=0.05,
        atol=0.01,
    )
