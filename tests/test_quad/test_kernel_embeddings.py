"""Test cases for kernel embeddings."""

from typing import Optional, Tuple, Union

import numpy as np
import pytest
from scipy.linalg import sqrtm
from scipy.special import roots_legendre

from probnum.typing import FloatArgType, IntArgType


# Auxiliary functions
def gauss_hermite_tensor(
    n_points: IntArgType,
    dim: IntArgType,
    mean: Union[np.ndarray, FloatArgType],
    cov: Union[np.ndarray, FloatArgType],
):
    """Returns the points and weights of a tensor-product Gauss-Hermite rule for
    integration w.r.t a Gaussian measure."""
    x_gh_1d, w_gh = np.polynomial.hermite.hermgauss(n_points)
    x_gh = (
        np.sqrt(2)
        * np.stack(np.meshgrid(*(x_gh_1d,) * dim), -1).reshape(-1, dim)
        @ sqrtm(np.atleast_2d(cov))
        + mean
    )
    w_gh = np.prod(
        np.stack(np.meshgrid(*(w_gh,) * dim), -1).reshape(-1, dim), axis=1
    ) / (np.pi ** (dim / 2))
    return x_gh, w_gh


def gauss_legendre_tensor(
    n_points: IntArgType,
    dim: IntArgType,
    domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
    normalized: Optional[bool] = False,
):
    """Returns the points and weights of a tensor-product Gauss-Legendre rule for
    integration w.r.t the Lebesgue measure on a hyper-rectangle."""
    x_1d, w_gl, weight_sum = roots_legendre(n_points, True)
    x_1d_shifted = [
        0.5 * (x_1d * (domain[1][i] - domain[0][i]) + domain[1][i] + domain[0][i])
        for i in range(0, dim)
    ]
    x_gl = np.stack(np.meshgrid(*x_1d_shifted), -1).reshape(-1, dim)
    w_gl = (
        np.prod(np.stack(np.meshgrid(*(w_gl,) * dim), -1).reshape(-1, dim), axis=1)
        / weight_sum ** dim
    )
    if not normalized:
        w_gl = w_gl * np.prod(domain[1] - domain[0])
    return x_gl, w_gl


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
        dim=kernel_embedding.dim,
        mean=kernel_embedding.measure.mean,
        cov=kernel_embedding.measure.cov,
    )

    x = kernel_embedding.measure.sample(rng, num_data)
    num_kernel_means = kernel_embedding.kernel(x, x_gh) @ w_gh
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
        dim=kernel_embedding.dim,
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
        dim=kernel_embedding.dim,
        domain=kernel_embedding.measure.domain,
        normalized=kernel_embedding.measure.normalized,
    )

    x = kernel_embedding.measure.sample(rng, num_data)
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
    n_gl = 20
    x_gl, w_gl = gauss_legendre_tensor(
        n_points=n_gl,
        dim=kernel_embedding.dim,
        domain=kernel_embedding.measure.domain,
        normalized=kernel_embedding.measure.normalized,
    )
    num_kernel_variance = kernel_embedding.kernel_mean(x_gl) @ w_gl
    true_kernel_variance = kernel_embedding.kernel_variance()
    np.testing.assert_allclose(
        true_kernel_variance, num_kernel_variance, rtol=1.0e-3, atol=1.0e-3
    )
