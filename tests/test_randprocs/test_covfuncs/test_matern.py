"""Test cases for the Matérn covariance function."""

import numpy as np
import pytest

from probnum.randprocs import covfuncs
from probnum.randprocs.covfuncs._matern import _matern_bessel
from probnum.typing import ShapeType


@pytest.mark.parametrize("nu", [-1, -1.0, 0.0, 0])
def test_nonpositive_nu_raises_exception(nu):
    """Check whether a non-positive nu parameter raises a ValueError."""
    with pytest.raises(ValueError):
        covfuncs.Matern(input_shape=(), nu=nu)


@pytest.mark.parametrize("lengthscales", (0.0, -1.0, (0.0, 1.0), (-0.2, 2.0)))
def test_nonpositive_lengthscales_raises_exception(lengthscales):
    """Check whether a non-positive `lengthscales` parameter raises a ValueError."""
    with pytest.raises(ValueError):
        covfuncs.Matern(np.shape(lengthscales), lengthscales=lengthscales)


@pytest.mark.parametrize("p", range(4))
def test_half_integer_impl_equals_naive_impl(
    input_shape: ShapeType, p: int, x0: np.ndarray, x1: np.ndarray
):
    """Test whether the optimized half-integer implementation produces the same results
    as the general implementation based on the Bessel function."""
    rng = np.random.default_rng(537892325)

    nu = p + 0.5
    lengthscales = rng.gamma(1.0, size=input_shape) + 0.5

    k = covfuncs.Matern(input_shape, nu=nu, lengthscales=lengthscales)

    assert k.is_half_integer
    assert k.p == p

    assert np.all(lengthscales > 0)

    # Compare against naive implementation
    k_naive = NaiveMatern(input_shape, nu=nu, lengthscales=lengthscales)

    np.testing.assert_allclose(k.matrix(x0, x1), k_naive.matrix(x0, x1))


def test_nu_large_recovers_rbf_covfunc(
    x0: np.ndarray, x1: np.ndarray, input_shape: ShapeType
):
    """Test whether a Matern covariance function with nu large is close to an RBF
    covariance function."""
    lengthscale = 1.25
    rbf = covfuncs.ExpQuad(input_shape=input_shape, lengthscales=lengthscale)
    matern = covfuncs.Matern(input_shape=input_shape, lengthscales=lengthscale, nu=15)

    np.testing.assert_allclose(
        rbf.matrix(x0, x1),
        matern.matrix(x0, x1),
        err_msg=(
            "RBF and Matern covariance function are not sufficiently close for "
            "nu->infty."
        ),
        rtol=0.05,
        atol=0.01,
    )


class NaiveMatern(covfuncs.IsotropicMixin, covfuncs.CovarianceFunction):
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
