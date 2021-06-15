"""Auxiliary functions used in tests for quadrature methods."""
from typing import Optional, Tuple, Union

import numpy as np
from scipy.linalg import sqrtm
from scipy.special import roots_legendre

from probnum.type import FloatArgType, IntArgType


# Auxiliary functions for quadrature tests
def gauss_hermite_tensor(
    n_points: IntArgType,
    input_dim: IntArgType,
    mean: Union[np.ndarray, FloatArgType],
    cov: Union[np.ndarray, FloatArgType],
):
    """Returns the points and weights of a tensor-product Gauss-Hermite rule for
    integration w.r.t a Gaussian measure."""
    x_gh_1d, w_gh = np.polynomial.hermite.hermgauss(n_points)
    x_gh = (
        np.sqrt(2)
        * np.stack(np.meshgrid(*(x_gh_1d,) * input_dim), -1).reshape(-1, input_dim)
        @ sqrtm(np.atleast_2d(cov))
        + mean
    )
    w_gh = np.prod(
        np.stack(np.meshgrid(*(w_gh,) * input_dim), -1).reshape(-1, input_dim), axis=1
    ) / (np.pi ** (input_dim / 2))
    return x_gh, w_gh


def gauss_legendre_tensor(
    n_points: IntArgType,
    input_dim: IntArgType,
    domain: Tuple[Union[np.ndarray, FloatArgType], Union[np.ndarray, FloatArgType]],
    normalized: Optional[bool] = False,
):
    """Returns the points and weights of a tensor-product Gauss-Legendre rule for
    integration w.r.t the Lebesgue measure on a hyper-rectangle."""
    x_1d, w_gl, weight_sum = roots_legendre(n_points, True)
    x_1d_shifted = [
        0.5 * (x_1d * (domain[1][i] - domain[0][i]) + domain[1][i] + domain[0][i])
        for i in range(0, input_dim)
    ]
    x_gl = np.stack(np.meshgrid(*x_1d_shifted), -1).reshape(-1, input_dim)
    w_gl = (
        np.prod(
            np.stack(np.meshgrid(*(w_gl,) * input_dim), -1).reshape(-1, input_dim),
            axis=1,
        )
        / weight_sum ** input_dim
    )
    if not normalized:
        w_gl = w_gl * np.prod(domain[1] - domain[0])
    return x_gl, w_gl
