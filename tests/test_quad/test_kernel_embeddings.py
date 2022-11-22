"""Test cases for kernel embeddings."""

import numpy as np
from scipy.integrate import quad

from probnum.quad.kernel_embeddings import KernelEmbedding

from .util import gauss_hermite_tensor, gauss_legendre_tensor

import pytest


# Common tests
def test_kernel_mean_shape(kernel_embedding, x):
    """Test output shape of kernel mean."""

    kernel_mean_shape = (np.atleast_2d(x).shape[0],)

    assert kernel_embedding.kernel_mean(x).shape == kernel_mean_shape, (
        f"Kernel mean of {type(kernel_embedding)} has shape"
        f" {kernel_embedding.kernel_mean(x).shape} instead of {kernel_mean_shape}"
    )


def test_kernel_variance_float(kernel_embedding):
    """Test output of kernel variance."""
    assert isinstance(kernel_embedding.kernel_variance(), float)


# Tests for squared exponential kernel and Gaussian measure
@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["gauss"])
def test_kernel_mean_gaussian_measure(kernel_embedding, num_data, rng):
    """Test kernel means for the Gaussian measure against Gauss-Hermite tensor product
    rule."""
    n_gh = 10
    x_gh, w_gh = gauss_hermite_tensor(
        n_points=n_gh,
        input_dim=kernel_embedding.input_dim,
        mean=kernel_embedding.measure.mean,
        cov=kernel_embedding.measure.cov,
    )

    x = kernel_embedding.measure.sample(rng=rng, n_sample=num_data)
    num_kernel_means = kernel_embedding.kernel.matrix(x, x_gh) @ w_gh
    true_kernel_means = kernel_embedding.kernel_mean(x)
    np.testing.assert_allclose(
        true_kernel_means, num_kernel_means, rtol=1.0e-2, atol=1.0e-2
    )


@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["gauss"])
def test_kernel_var_gaussian_measure(kernel_embedding):
    """Test kernel variance for the Gaussian measure against Gauss-Hermite tensor
    product rule."""
    n_gh = 20
    x_gh, w_gh = gauss_hermite_tensor(
        n_points=n_gh,
        input_dim=kernel_embedding.input_dim,
        mean=kernel_embedding.measure.mean,
        cov=kernel_embedding.measure.cov,
    )

    num_kernel_variance = kernel_embedding.kernel_mean(x_gh) @ w_gh
    true_kernel_variance = kernel_embedding.kernel_variance()
    np.testing.assert_allclose(
        true_kernel_variance, num_kernel_variance, rtol=1.0e-3, atol=1.0e-3
    )


# Tests for squared exponential kernel and Lebesgue measure
@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_kernel_mean_lebesgue_measure(kernel_embedding, num_data, rng):
    """Test kernel means for the Lebesgue measure against Gauss-Legendre tensor product
    rule."""
    n_gl = 10
    x_gl, w_gl = gauss_legendre_tensor(
        n_points=n_gl,
        input_dim=kernel_embedding.input_dim,
        domain=kernel_embedding.measure.domain,
        normalized=kernel_embedding.measure.normalized,
    )

    x = kernel_embedding.measure.sample(rng=rng, n_sample=num_data)
    num_kernel_means = kernel_embedding.kernel.matrix(x, x_gl) @ w_gl
    true_kernel_means = kernel_embedding.kernel_mean(x)
    np.testing.assert_allclose(
        true_kernel_means, num_kernel_means, rtol=1.0e-3, atol=1.0e-3
    )


@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_kernel_var_lebesgue_measure(kernel_embedding):
    """Test kernel variance for the Lebesgue measure against Gauss-Legendre tensor
    product rule."""
    n_gl = 20
    x_gl, w_gl = gauss_legendre_tensor(
        n_points=n_gl,
        input_dim=kernel_embedding.input_dim,
        domain=kernel_embedding.measure.domain,
        normalized=kernel_embedding.measure.normalized,
    )
    num_kernel_variance = kernel_embedding.kernel_mean(x_gl) @ w_gl
    true_kernel_variance = kernel_embedding.kernel_variance()
    np.testing.assert_allclose(
        true_kernel_variance, num_kernel_variance, rtol=1.0e-3, atol=1.0e-3
    )


# Tests for Matern kernels and Lebesgue measure
@pytest.mark.parametrize("input_dim", [1, 2, 3])
@pytest.mark.parametrize("num_data", [1, 2, 5])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_kernel_mean_matern_lebesgue_measure(matern_kernel, measure, num_data, rng):
    kernel_embedding = KernelEmbedding(matern_kernel, measure)
    points = kernel_embedding.measure.sample(rng=rng, n_sample=num_data)
    # Numerical integration using scipy's quad
    num_kernel_means = np.zeros((num_data,))
    a, b = kernel_embedding.measure.domain
    for indx in range(num_data):
        integral_current_dim = 1.0
        for dim in range(kernel_embedding.measure.input_dim):

            def fun(x, d=dim, ind=indx):
                return kernel_embedding.kernel.univariate_materns[d](points[ind][d], x)

            integral_current_dim *= quad(fun, a[dim], b[dim])[0]
        num_kernel_means[indx] = (
            integral_current_dim * kernel_embedding.measure.normalization_constant
        )
    # True kernel means
    true_kernel_means = kernel_embedding.kernel_mean(points)
    # Compare
    np.testing.assert_allclose(
        true_kernel_means, num_kernel_means, rtol=1.0e-5, atol=1.0e-5
    )


@pytest.mark.parametrize("input_dim", [1, 2, 3])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_kernel_variance_matern_lebesgue_measure(matern_kernel, measure):
    kernel_embedding = KernelEmbedding(matern_kernel, measure)
    n_gl = 20
    x_gl, w_gl = gauss_legendre_tensor(
        n_points=n_gl,
        input_dim=kernel_embedding.input_dim,
        domain=kernel_embedding.measure.domain,
        normalized=kernel_embedding.measure.normalized,
    )
    num_kernel_variance = kernel_embedding.kernel_mean(x_gl) @ w_gl
    true_kernel_variance = kernel_embedding.kernel_variance()
    np.testing.assert_allclose(
        true_kernel_variance, num_kernel_variance, rtol=1.0e-3, atol=1.0e-3
    )
