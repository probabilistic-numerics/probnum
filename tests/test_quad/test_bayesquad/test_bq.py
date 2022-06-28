"""Test cases for Bayesian quadrature."""

import numpy as np
import pytest
from scipy.integrate import quad

import probnum.quad
from probnum.quad import bayesquad, bayesquad_from_data
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.randvars import Normal

from ..util import gauss_hermite_tensor, gauss_legendre_tensor


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.mark.parametrize("input_dim", [1], ids=["dim1"])
def test_type_1d(f1d, kernel, measure, input_dim):
    """Test that BQ outputs normal random variables for 1D integrands."""
    integral, _ = bayesquad(
        fun=f1d, input_dim=input_dim, kernel=kernel, measure=measure, max_evals=10
    )
    assert isinstance(integral, Normal)


@pytest.mark.parametrize("input_dim", [1])
@pytest.mark.parametrize(
    "domain",
    [
        (
            0,
            1,
        ),
        (-0.5, 1),
        (-3.5, -2.9),
    ],
)
@pytest.mark.parametrize("var_tol", [None, 1e-7])
@pytest.mark.parametrize("rel_tol", [None, 1e-7])
@pytest.mark.parametrize("scale_estimation", [None, "mle"])
@pytest.mark.parametrize("jitter", [1e-6, 1e-7])
def test_integral_values_1d(
    f1d, kernel, domain, input_dim, scale_estimation, var_tol, rel_tol, jitter
):
    """Test numerically that BQ computes 1D integrals correctly for a number of
    different parameters."""

    measure = probnum.quad.LebesgueMeasure(input_dim=input_dim, domain=domain)
    # numerical integral
    # pylint: disable=invalid-name
    def integrand(x):
        return f1d(x) * measure(np.atleast_2d(x))

    # pylint: disable=invalid-name
    bq_integral, _ = bayesquad(
        fun=f1d,
        input_dim=input_dim,
        kernel=kernel,
        domain=domain,
        policy="vdc",
        scale_estimation=scale_estimation,
        max_evals=250,
        var_tol=var_tol,
        rel_tol=rel_tol,
        jitter=jitter,
    )
    domain = measure.domain
    if domain is None:
        domain = (-np.infty, np.infty)
    num_integral, _ = quad(integrand, domain[0], domain[1])
    np.testing.assert_almost_equal(bq_integral.mean, num_integral, decimal=2)


@pytest.mark.parametrize("input_dim", [2, 3, 4])
@pytest.mark.parametrize("measure_name", ["gauss"])
@pytest.mark.parametrize("cov_diagonal", [True])
@pytest.mark.parametrize("scale_estimation", [None, "mle"])
def test_integral_values_x2_gaussian(kernel, measure, input_dim, scale_estimation):
    """Test numerical integration of x**2 in higher dimensions."""
    # pylint: disable=invalid-name
    c = np.linspace(0.1, 2.2, input_dim)
    fun = lambda x: np.sum(c * x**2, 1)
    true_integral = np.sum(c * (measure.mean**2 + np.diag(measure.cov)))
    n_gh = 8  # Be very careful about increasing this - yields huge kernel matrices
    nodes, _ = gauss_hermite_tensor(
        n_points=n_gh, input_dim=input_dim, mean=measure.mean, cov=measure.cov
    )
    fun_evals = fun(nodes)
    bq_integral, _ = bayesquad_from_data(
        nodes=nodes,
        fun_evals=fun_evals,
        kernel=kernel,
        measure=measure,
        scale_estimation=scale_estimation,
    )
    np.testing.assert_almost_equal(bq_integral.mean, true_integral, decimal=2)


@pytest.mark.parametrize("input_dim", [2, 3, 4])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
@pytest.mark.parametrize("scale_estimation", [None, "mle"])
@pytest.mark.parametrize("jitter", [1e-6, 0.5e-5])
def test_integral_values_sin_lebesgue(
    kernel, measure, input_dim, scale_estimation, jitter
):
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
        input_dim=input_dim,
        domain=measure.domain,
        normalized=measure.normalized,
    )
    fun_evals = fun(nodes)
    bq_integral, _ = bayesquad_from_data(
        nodes=nodes,
        fun_evals=fun_evals,
        kernel=kernel,
        measure=measure,
        scale_estimation=scale_estimation,
        jitter=jitter,
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
            max_evals=1000,
            batch_size=50,
        )
        true_integral = kernel_embedding.kernel_mean(np.atleast_2d(translate_point))
        np.testing.assert_almost_equal(bq_integral.mean, true_integral, decimal=2)


def test_no_domain_or_measure_raises_error(input_dim):
    """Test that errors are correctly raised when both domain and a Gaussian measure is
    given."""
    fun = lambda x: np.ones(x.shape[0])
    nodes = np.linspace(0, 1, 3)
    fun_evals = fun(nodes)

    with pytest.raises(ValueError):
        bayesquad(fun=fun, input_dim=input_dim)

    with pytest.raises(ValueError):
        bayesquad_from_data(nodes=nodes, fun_evals=fun_evals)


@pytest.mark.parametrize("input_dim", [1])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_domain_ignored_if_lebesgue(input_dim, measure):
    domain = (0, 1)
    fun = lambda x: np.reshape(x, (x.shape[0],))

    # standard BQ
    bq_integral, _ = bayesquad(
        fun=fun, input_dim=input_dim, domain=domain, measure=measure
    )
    assert isinstance(bq_integral, Normal)

    # fixed nodes BQ
    nodes = np.linspace(0, 1, 3).reshape((3, 1))
    fun_evals = fun(nodes)

    bq_integral, _ = bayesquad_from_data(
        nodes=nodes, fun_evals=fun_evals, domain=domain, measure=measure
    )
    assert isinstance(bq_integral, Normal)


@pytest.mark.parametrize("jitter", [-1.0, -0.002])
def test_negative_jitter_throws_error(jitter):
    """Test that negative values for jitter raise ValueError."""
    input_dim = 1
    domain = (0, 1)
    fun = lambda x: np.reshape(x, (x.shape[0],))
    with pytest.raises(ValueError):
        bayesquad(fun=fun, input_dim=input_dim, domain=domain, jitter=jitter)


def test_zero_function_gives_zero_variance_with_mle():
    """Test that BQ variance is zero for zero function when MLE is used to set the
    scale parameter."""
    input_dim = 1
    domain = (0, 1)
    fun = lambda x: np.zeros(x.shape[0])
    nodes = np.linspace(0, 1, 3).reshape((3, 1))
    fun_evals = fun(nodes)

    bq_integral1, _ = bayesquad(
        fun=fun, input_dim=input_dim, domain=domain, scale_estimation="mle"
    )
    bq_integral2, _ = bayesquad_from_data(
        nodes=nodes, fun_evals=fun_evals, domain=domain, scale_estimation="mle"
    )
    assert bq_integral1.var == 0.0
    assert bq_integral2.var == 0.0
