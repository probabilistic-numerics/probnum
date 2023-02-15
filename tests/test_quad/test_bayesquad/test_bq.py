"""Test cases for Bayesian quadrature."""
import copy

import numpy as np
import pytest
from scipy.integrate import quad as scipyquad

from probnum.quad import bayesquad, bayesquad_from_data, multilevel_bayesquad_from_data
from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.randvars import Normal

from ..util import gauss_hermite_tensor, gauss_legendre_tensor


@pytest.mark.parametrize("input_dim", [1], ids=["dim1"])
def test_type_1d(f1d, kernel, measure, input_dim, rng):
    """Test that BQ outputs normal random variables for 1D integrands."""
    integral, _ = bayesquad(
        fun=f1d,
        input_dim=input_dim,
        kernel=kernel,
        measure=measure,
        rng=rng,
        options=dict(max_evals=10),
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
    f1d, kernel, domain, input_dim, scale_estimation, var_tol, rel_tol, jitter, rng
):
    """Test numerically that BQ computes 1D integrals correctly for a number of
    different parameters.

    The test currently uses van der Corput policy and therefore works only for finite
    domains.
    """

    measure = LebesgueMeasure(input_dim=input_dim, domain=domain)
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
        rng=None,
        options=dict(
            scale_estimation=scale_estimation,
            max_evals=250,
            var_tol=var_tol,
            rel_tol=rel_tol,
            jitter=jitter,
        ),
    )
    domain = measure.domain
    num_integral, _ = scipyquad(integrand, domain[0], domain[1])
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
        options=dict(scale_estimation=scale_estimation),
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
        options=dict(scale_estimation=scale_estimation, jitter=jitter),
    )
    np.testing.assert_almost_equal(bq_integral.mean, true_integral, decimal=2)


@pytest.mark.parametrize("input_dim", [2, 3, 4])
@pytest.mark.parametrize("num_data", [1])
# pylint: disable=invalid-name
def test_integral_values_kernel_translate(kernel, measure, input_dim, x, rng):
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
            rng=rng,
            options=dict(max_evals=1000, var_tol=1e-8, batch_size=50),
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
def test_domain_ignored_if_lebesgue(input_dim, measure, rng):
    domain = (0, 1)
    fun = lambda x: np.reshape(x, (x.shape[0],))

    # standard BQ
    bq_integral, _ = bayesquad(
        fun=fun, input_dim=input_dim, domain=domain, measure=measure, rng=rng
    )
    assert isinstance(bq_integral, Normal)

    # fixed nodes BQ
    nodes = np.linspace(0, 1, 3).reshape((3, 1))
    fun_evals = fun(nodes)

    bq_integral, _ = bayesquad_from_data(
        nodes=nodes, fun_evals=fun_evals, domain=domain, measure=measure
    )
    assert isinstance(bq_integral, Normal)


def test_zero_function_gives_zero_variance_with_mle(rng):
    """Test that BQ variance is zero for zero function when MLE is used to set the
    scale parameter."""
    input_dim = 1
    domain = (0, 1)
    fun = lambda x: np.zeros(x.shape[0])
    nodes = np.linspace(0, 1, 3).reshape((3, 1))
    fun_evals = fun(nodes)

    bq_integral1, _ = bayesquad(
        fun=fun,
        input_dim=input_dim,
        domain=domain,
        rng=rng,
        options=dict(scale_estimation="mle"),
    )
    bq_integral2, _ = bayesquad_from_data(
        nodes=nodes,
        fun_evals=fun_evals,
        domain=domain,
        options=dict(scale_estimation="mle"),
    )
    assert bq_integral1.var == 0.0
    assert bq_integral2.var == 0.0


def test_multilevel_bayesquad_from_data_input_handling(kernel, measure, rng):
    """Test that inputs to multilevel BQ are handled properly."""
    n_level = 3
    ns_1 = (3, 7, 2)
    fun_diff_evals_1 = tuple([np.zeros(ns_1[l]) for l in range(n_level)])
    nodes_full = tuple([measure.sample((ns_1[l]), rng=rng) for l in range(n_level)])

    F, infos = multilevel_bayesquad_from_data(
        nodes=nodes_full,
        fun_diff_evals=fun_diff_evals_1,
        measure=measure,
    )
    assert isinstance(F, Normal)
    assert len(infos) == n_level
    # Only one set of nodes
    kernels_1 = tuple(copy.deepcopy(kernel) for l in range(n_level))
    ns_2 = (7, 7, 7)
    fun_diff_evals_2 = n_level * (np.zeros((ns_2[0],)),)
    kernels_full = n_level * (kernel,)
    nodes_1 = (measure.sample(n_sample=ns_2[0], rng=rng),)
    F, infos = multilevel_bayesquad_from_data(
        nodes=nodes_1,
        fun_diff_evals=fun_diff_evals_2,
        kernels=kernels_full,
        measure=measure,
    )
    assert isinstance(F, Normal)
    assert len(infos) == n_level
    # Only one kernel and one set of nodes
    F, infos = multilevel_bayesquad_from_data(
        nodes=nodes_1,
        fun_diff_evals=fun_diff_evals_2,
        kernels=kernels_1,
        measure=measure,
    )
    assert isinstance(F, Normal)
    assert len(infos) == n_level
    # Wrong number inputs should throw error
    kernels_2 = (kernel, kernel)
    with pytest.raises(ValueError):
        _, _ = multilevel_bayesquad_from_data(
            nodes=nodes_1,
            fun_diff_evals=fun_diff_evals_2,
            kernels=kernels_2,
            measure=measure,
        )
    nodes_2 = (nodes_full[0], nodes_full[1])
    with pytest.raises(ValueError):
        _, _ = multilevel_bayesquad_from_data(
            nodes=nodes_2,
            fun_diff_evals=fun_diff_evals_2,
            kernels=kernels_2,
            measure=measure,
        )


def test_multilevel_bayesquad_from_data_equals_bq_with_trivial_data_1d():
    """Test that multilevel BQ equals BQ when all but one level are given non-zero
    function evaluations for 1D data."""
    input_dim = 1
    n_level = 5
    domain = (0, 3.3)
    nodes = ()
    nodes = [np.linspace(0, 1, 2 * l + 1)[:, None] for l in range(n_level)]
    for i in range(n_level):
        jitter = 1e-5 * (i + 1.0)
        fun_diff_evals = [np.zeros(shape=(len(xs),)) for xs in nodes]
        fun_evals = nodes[i][:, 0] ** (2 + 0.3 * i) + 1.2
        fun_diff_evals[i] = fun_evals
        mlbq_integral, _ = multilevel_bayesquad_from_data(
            nodes=tuple(nodes),
            fun_diff_evals=tuple(fun_diff_evals),
            domain=domain,
            options=dict(jitter=jitter),
        )
        bq_integral, _ = bayesquad_from_data(
            nodes=nodes[i],
            fun_evals=fun_evals,
            domain=domain,
            options=dict(jitter=jitter),
        )
        assert mlbq_integral.mean == bq_integral.mean
        assert mlbq_integral.cov == bq_integral.cov


def test_multilevel_bayesquad_from_data_equals_bq_with_trivial_data_2d():
    """Test that multilevel BQ equals BQ when all but one level are given non-zero
    function evaluations for 2D data."""
    input_dim = 2
    n_level = 5
    measure = GaussianMeasure(np.full((input_dim,), 0.2), cov=0.6 * np.eye(input_dim))
    _gh = gauss_hermite_tensor
    nodes = [_gh(l + 1, input_dim, measure.mean, measure.cov)[0] for l in range(n_level)]
    for i in range(n_level):
        jitter = 1e-5 * (i + 1.0)
        fun_diff_evals = [np.zeros(shape=(len(xs),)) for xs in nodes]
        fun_evals =  np.sin(nodes[i][:, 0] * i) + (i + 1.0) * np.cos(nodes[i][:, 1])
        fun_diff_evals[i] = fun_evals
        mlbq_integral, _ = multilevel_bayesquad_from_data(
            nodes=tuple(nodes),
            fun_diff_evals=tuple(fun_diff_evals),
            measure=measure,
            options=dict(jitter=jitter),
        )
        bq_integral, _ = bayesquad_from_data(
            nodes=nodes[i],
            fun_evals=fun_evals,
            measure=measure,
            options=dict(jitter=jitter),
        )
        assert mlbq_integral.mean == bq_integral.mean
        assert mlbq_integral.cov == bq_integral.cov
