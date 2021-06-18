"""Product Matern kernel."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.type import IntArgType

from ._kernel import Kernel
from ._matern import Matern

_InputType = np.ndarray


class ProductMatern(Kernel[_InputType]):
    """Product Matern kernel.

    Covariance function defined as a product of one-dimensional Matern
    kernels: :math:`k(x_0, x_1) = \prod_{i=1}^d k_i(x_{0,i}, x_{1,i}`,
    where :math:`x_0 = (x_{0,i}, \ldots, x_{0,d})` and :math:`x_0 = (x_{0,i}, \ldots, x_{0,d})`
    and :math:`k_i` are one-dimensional Matern kernels.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    lengthscales :
        Lengthscales of the one-dimensional Matern kernels. Describes the input scale on
        which the process varies.
    nus :
        Hyperparameters controlling differentiability of the one-dimensional Matern
        kernels.

    See Also
    --------
    Matern : Stationary Matern kernel.
    """

    def __init__(
        self, input_dim: IntArgType, lengthscales: np.ndarray, nus: np.ndarray
    ):
        # If only single lengthcsale or nu is given, use this in every dimension
        if np.isscalar(lengthscales) or lengthscales.size == 1:
            lengthscales = np.full((input_dim,), lengthscales)
        if np.isscalar(nus) or nus.size == 1:
            nus = np.full((input_dim,), nus)

        one_d_materns = []
        for dim in range(input_dim):
            one_d_materns.append(
                Matern(input_dim=1, lengthscale=lengthscales[dim], nu=nus[dim])
            )
        self.one_d_materns = one_d_materns
        self.nus = nus
        self.lengthscales = lengthscales

        super().__init__(input_dim=input_dim, output_dim=1)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:

        x0, x1, kernshape = self._check_and_reshape_inputs(x0, x1)
        kernmat = np.ones(kernshape)

        if x1 is None:
            for dim in range(self.input_dim):
                kernmat *= self.one_d_materns[dim](_utils.as_colvec(x0[:, dim]))
        else:
            for dim in range(self.input_dim):
                kernmat *= self.one_d_materns[dim](
                    _utils.as_colvec(x0[:, dim]), _utils.as_colvec(x1[:, dim])
                )

        return Kernel._reshape_kernelmatrix(kernmat, newshape=kernshape)
