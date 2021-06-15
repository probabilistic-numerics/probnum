"""Test cases for Bayesian quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

from probnum.quad import bayesquad, bayesquad_fixed
from probnum.quad.kernel_embeddings._kernel_embedding import KernelEmbedding
from probnum.randvars import Normal

from ..aux import gauss_hermite_tensor, gauss_legendre_tensor


@pytest.mark.parametrize("input_dim", [1], ids=["dim1"])
def test_type_1d(f1d, kernel, measure, input_dim):
    """Test that BQ outputs normal random variables for 1D integrands."""
    integral, _ = bayesquad(
        fun=f1d, input_dim=input_dim, kernel=kernel, measure=measure, max_nevals=10
    )
    assert isinstance(integral, Normal)


@pytest.mark.parametrize("input_dim", [1])
def test_integral_values_1d(f1d, kernel, measure, input_dim):
    """Test numerically that BQ computes 1D integrals correctly."""

    # numerical integral
    # pylint: disable=invalid-name
    def integrand(x):
        return f1d(x) * measure(x)

    # pylint: disable=invalid-name
    bq_integral, _ = bayesquad(
        fun=f1d, input_dim=input_dim, kernel=kernel, measure=measure, max_nevals=250
    )
    num_integral, _ = quad(integrand, measure.domain[0], measure.domain[1])
    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)


@pytest.mark.parametrize("input_dim", [2, 3, 4])
@pytest.mark.parametrize("measure_name", ["gauss"])
@pytest.mark.parametrize("cov_diagonal", [True])
def test_integral_values_x2_gaussian(kernel, measure, input_dim):
    """Test numerical integration of x**2 in higher dimensions."""
    # pylint: disable=invalid-name
    c = np.linspace(0.1, 2.2, input_dim)
    fun = lambda x: np.sum(c * x ** 2, 1)
    true_integral = np.sum(c * (measure.mean ** 2 + np.diag(measure.cov)))
    n_gh = 8  # Be very careful about increasing this - yields huge kernel matrices
    nodes, _ = gauss_hermite_tensor(
        n_points=n_gh, dim=input_dim, mean=measure.mean, cov=measure.cov
    )
    fun_evals = fun(nodes)
    bq_integral, _ = bayesquad_fixed(
        nodes=nodes, f_evals=fun_evals, kernel=kernel, measure=measure
    )
    np.testing.assert_almost_equal(bq_integral.mean, true_integral, decimal=2)


@pytest.mark.parametrize("input_dim", [2, 3, 4])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_integral_values_sin_lebesgue(kernel, measure, input_dim):
    """Test numerical integration of products of sinusoids."""
    # pylint: disable=invalid-name
    c = np.linspace(0.1, 0.5, input_dim)
    (a, b) = measure.domain
    fun = lambda x: np.prod(np.sin(c * x), 1)
    true_integral = (
        np.prod((np.cos(c * a) - np.cos(c * b)) / c) * measure.normalization_constant
    )
    n_gl = 8  # Be very careful about increasing this - yields huge kernel matrices
    nodes, _ = gauss_legendre_tensor(
        n_points=n_gl,
        dim=input_dim,
        domain=measure.domain,
        normalized=measure.normalized,
    )
    fun_evals = fun(nodes)
    bq_integral, _ = bayesquad_fixed(
        nodes=nodes, f_evals=fun_evals, kernel=kernel, measure=measure
    )
    np.testing.assert_almost_equal(bq_integral.mean, true_integral, decimal=2)


@pytest.mark.parametrize("input_dim", [2, 3, 4])
@pytest.mark.parametrize("num_data", [1])
# pylint: disable=invalid-name
def test_integral_values_kernel_translate(kernel, measure, input_dim, x):
    """Test numerical integration of kernel translates."""
    kernel_embedding = KernelEmbedding(kernel, measure)
    # pylint: disable=cell-var-from-loop
    for translate_point in x:
        fun = lambda y: kernel(translate_point, y)
        bq_integral, _ = bayesquad(
            fun=fun,
            input_dim=input_dim,
            kernel=kernel,
            measure=measure,
            var_tol=1e-8,
            max_nevals=1000,
            batch_size=50,
        )
        true_integral = kernel_embedding.kernel_mean(np.atleast_2d(translate_point))
        np.testing.assert_almost_equal(bq_integral.mean, true_integral, decimal=2)
