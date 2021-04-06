"""Test cases for kernel embeddings."""

import numpy as np
import pytest
from scipy.linalg import sqrtm
from scipy.special import roots_legendre


def test_kmean_shape(kernel_embedding, x):
    """Test output shape of kernel mean."""

    kmean_shape = (np.atleast_2d(x).shape[0],)

    assert kernel_embedding.kernel_mean(x).shape == kmean_shape, (
        f"Kernel mean of {type(kernel_embedding)} has shape {kernel_embedding.kernel_mean(x).shape} instead of"
        f" {kmean_shape}"
    )


def test_kvar_float(kernel_embedding):
    """Test output of kernel variance."""
    assert isinstance(kernel_embedding.kernel_variance(), np.float)


@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["gauss"])
def test_kernel_mean_gaussian_measure(kernel_embedding, x_gauss):
    """Test kernel means for the Gaussian measure against Gauss-Hermite tensor product
    rule."""
    dim = kernel_embedding.dim
    n_gh = 10
    x_gh_1d, w_gh = np.polynomial.hermite.hermgauss(n_gh)
    x_gh = (
        np.sqrt(2)
        * np.stack(np.meshgrid(*(x_gh_1d,) * dim), -1).reshape(-1, dim)
        @ sqrtm(np.atleast_2d(kernel_embedding.measure.cov))
        + kernel_embedding.measure.mean
    )
    w_gh = np.prod(
        np.stack(np.meshgrid(*(w_gh,) * dim), -1).reshape(-1, dim), axis=1
    ) / (np.pi ** (dim / 2))

    num_kernel_means = kernel_embedding.kernel(x_gauss, x_gh) @ w_gh
    true_kernel_means = kernel_embedding.kernel_mean(x_gauss)
    np.testing.assert_allclose(
        true_kernel_means, num_kernel_means, rtol=1.0e-2, atol=1.0e-2
    )


@pytest.mark.parametrize("input_dim", [1, 2, 3])
@pytest.mark.parametrize("measure_name", ["gauss"])
def test_kernel_var_gaussian_measure(kernel_embedding):
    """Test kernel variance for the Gaussian measure against Gauss-Hermite tensor
    product rule."""
    dim = kernel_embedding.dim
    n_gh = 20
    x_gh_1d, w_gh = np.polynomial.hermite.hermgauss(n_gh)
    x_gh = (
        np.sqrt(2)
        * np.stack(np.meshgrid(*(x_gh_1d,) * dim), -1).reshape(-1, dim)
        @ sqrtm(np.atleast_2d(kernel_embedding.measure.cov))
        + kernel_embedding.measure.mean
    )
    w_gh = np.prod(
        np.stack(np.meshgrid(*(w_gh,) * dim), -1).reshape(-1, dim), axis=1
    ) / (np.pi ** (dim / 2))

    num_kernel_variance = kernel_embedding.kernel_mean(x_gh) @ w_gh
    true_kernel_variance = kernel_embedding.kernel_variance()
    np.testing.assert_allclose(
        true_kernel_variance, num_kernel_variance, rtol=1.0e-3, atol=1.0e-3
    )


@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_kernel_mean_lebesgue_measure(kernel_embedding, x):
    """Test kernel means for the Lebesgue measure against Gauss-Legendre tensor product
    rule."""
    dim = kernel_embedding.dim
    a, b = kernel_embedding.measure.domain[0], kernel_embedding.measure.domain[1]
    n_gl = 10
    x_gl_1d, w_gl = roots_legendre(n_gl)
    foo = [0.5 * (x_gl_1d * (b[i] - a[i]) + b[i] + a[i]) for i in range(0, dim)]
    x_gl = np.stack(np.meshgrid(*foo), -1).reshape(-1, dim)
    w_gl = (
        np.prod(np.stack(np.meshgrid(*(w_gl,) * dim), -1).reshape(-1, dim), axis=1)
        / 2 ** dim
    )

    num_kernel_means = kernel_embedding.kernel(x, x_gl) @ w_gl
    true_kernel_means = kernel_embedding.kernel_mean(x)
    np.testing.assert_allclose(
        true_kernel_means, num_kernel_means, rtol=1.0e-3, atol=1.0e-3
    )


@pytest.mark.parametrize("input_dim", [1, 2, 3, 5])
@pytest.mark.parametrize("measure_name", ["lebesgue"])
def test_kernel_var_lebesgue_measure(kernel_embedding):
    """Test kernel variance for the Lebesgue measure against Gauss-Legendre tensor
    product rule."""
    dim = kernel_embedding.dim
    a, b = kernel_embedding.measure.domain[0], kernel_embedding.measure.domain[1]
    n_gl = 20
    x_gl_1d, w_gl = roots_legendre(n_gl)
    foo = [0.5 * (x_gl_1d * (b[i] - a[i]) + b[i] + a[i]) for i in range(0, dim)]
    x_gl = np.stack(np.meshgrid(*foo), -1).reshape(-1, dim)
    w_gl = (
        np.prod(np.stack(np.meshgrid(*(w_gl,) * dim), -1).reshape(-1, dim), axis=1)
        / 2 ** dim
    )

    num_kernel_variance = kernel_embedding.kernel_mean(x_gl) @ w_gl
    true_kernel_variance = kernel_embedding.kernel_variance()
    np.testing.assert_allclose(
        true_kernel_variance, num_kernel_variance, rtol=1.0e-3, atol=1.0e-3
    )
