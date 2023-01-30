"""Test cases for the Matern kernel."""

import numpy as np
import pytest

from probnum.randprocs import kernels
from probnum.randprocs.kernels._matern import _matern_bessel
from probnum.typing import ShapeType


@pytest.mark.parametrize("p", range(4))
def test_half_integer_impl_equals_naive_impl(
    input_shape: ShapeType, p: int, x0: np.ndarray, x1: np.ndarray
):
    rng = np.random.default_rng(537892325)

    nu = p + 0.5
    lengthscales = rng.gamma(1.0, size=input_shape) + 0.5

    k = kernels.Matern(input_shape, nu=nu, lengthscales=lengthscales)

    assert k.is_half_integer
    assert k.p == p

    # Compare against naive implementation
    k_naive = NaiveMatern(input_shape, nu=nu, lengthscales=lengthscales)

    np.testing.assert_allclose(k.matrix(x0, x1), k_naive.matrix(x0, x1))


@pytest.mark.parametrize("nu", [-1, -1.0, 0.0, 0])
def test_nonpositive_nu_raises_exception(nu):
    """Check whether a non-positive nu parameter raises a ValueError."""
    with pytest.raises(ValueError):
        kernels.Matern(input_shape=(), nu=nu)


def test_nu_large_recovers_rbf_kernel(
    x0: np.ndarray, x1: np.ndarray, input_shape: ShapeType
):
    """Test whether a Matern kernel with nu large is close to an RBF kernel."""
    lengthscale = 1.25
    rbf = kernels.ExpQuad(input_shape=input_shape, lengthscales=lengthscale)
    matern = kernels.Matern(input_shape=input_shape, lengthscales=lengthscale, nu=15)

    np.testing.assert_allclose(
        rbf.matrix(x0, x1),
        matern.matrix(x0, x1),
        err_msg="RBF and Matern kernel are not sufficiently close for nu->infty.",
        rtol=0.05,
        atol=0.01,
    )


class NaiveMatern(kernels.IsotropicMixin, kernels.Kernel):
    def __init__(self, input_shape, *, nu, lengthscales):
        super().__init__(input_shape=input_shape)

        self._nu = nu
        self._lengthscales = lengthscales

    def _evaluate(self, x0, x1):
        return _matern_bessel(
            np.sqrt(2 * self._nu)
            * self._euclidean_distances(x0, x1, scale_factors=1.0 / self._lengthscales),
            nu=self._nu,
        )
