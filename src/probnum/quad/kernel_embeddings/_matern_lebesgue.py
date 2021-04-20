"""Kernel embedding of Matern kernels with Lebesgue integration measure."""

from typing import Tuple, Union

import numpy as np

import probnum.utils as _utils
from probnum.kernels import Matern, ProductMatern
from probnum.quad._integration_measures import LebesgueMeasure
from probnum.type import ScalarArgType


def _kernel_mean_matern_lebesgue(
    x: np.ndarray, kernel: Union[Matern, ProductMatern], measure: LebesgueMeasure
) -> np.ndarray:

    # Convert ordinary Matern kernel to a product Matern kernel
    if isinstance(kernel, Matern):
        if kernel.input_dim > 1:
            raise NotImplementedError(
                "Kernel embeddings have been implemented only for "
                "product Matern kernels in dimensions higher than "
                "one."
            )
        else:
            kernel = ProductMatern(
                input_dim=kernel.input_dim,
                lengthscales=kernel.lengthscale,
                nus=kernel.nu,
            )

    # Compute kernel mean via a product of one-dimensional kernel means
    kernel_mean = np.ones((x.shape[0],))
    for dim in range(kernel.input_dim):
        kernel_mean *= _kernel_mean_matern_1d_lebesgue(
            x=x[:, dim],
            kernel=kernel.one_d_materns[dim],
            domain=(measure.domain[0][dim], measure.domain[1][dim]),
        )

    return kernel_mean


def _kernel_mean_matern_1d_lebesgue(x: np.ndarray, kernel: Matern, domain: Tuple):
    (a, b) = domain
    ell = kernel.lengthscale
    if kernel.nu == 0.5:
        return ell * (2.0 - np.exp((a - x) / ell) - np.exp((x - b) / ell))
    else:
        raise NotImplementedError
