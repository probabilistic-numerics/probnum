"""Product Matern kernel."""

from typing import Optional, Union

import numpy as np

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import Kernel
from ._matern import Matern


class ProductMatern(Kernel):
    r"""Product Matern kernel.

    Covariance function defined as a product of one-dimensional Matern
    kernels: :math:`k(x_0, x_1) = \\prod_{i=1}^d k_i(x_{0,i}, x_{1,i})`,
    where :math:`x_0 = (x_{0,i}, \\ldots, x_{0,d})` and :math:`x_0 = (x_{0,i}, \\ldots,
    x_{0,d})` and :math:`k_i` are one-dimensional Matern kernels.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    lengthscales :
        Lengthscales of the one-dimensional Matern kernels. Describes the input scale on
        which the process varies. If a scalar, the same lengthscale is used in each
        dimension.
    nus :
        Hyperparameters controlling differentiability of the one-dimensional Matern
        kernels. If a scalar, the same smoothness is used in each dimension.

    See Also
    --------
    Matern : Stationary Matern kernel.
    ExpQuad : Exponentiated Quadratic / RBF kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.kernels import ProductMatern
    >>> input_dim = 2
    >>> lengthscales = np.array([0.1, 1.2])
    >>> nus = np.array([0.5, 3.5])
    >>> K = ProductMatern(input_dim=2, lengthscales=lenthscales, nus=nus)
    >>> xs = np.array([[0.0, 0.5], [1.0, 1.0], [0.5, 0.2]])
    >>> K.matrix(xs)
    array([[1.00000000e+00, 4.03712525e-05, 6.45332482e-03],
           [4.03712525e-05, 1.00000000e+00, 5.05119251e-03],
           [6.45332482e-03, 5.05119251e-03, 1.00000000e+00]])
    """

    def __init__(
        self,
        input_dim: IntArgType,
        lengthscales: Union[np.ndarray, ScalarArgType],
        nus: Union[np.ndarray, ScalarArgType],
    ):
        # If only single scalar lengthcsale or nu is given, use this in every dimension
        if np.isscalar(lengthscales):
            lengthscales = np.full((input_dim,), _utils.as_numpy_scalar(lengthscales))
        if np.isscalar(nus):
            nus = np.full((input_dim,), _utils.as_numpy_scalar(nus))

        one_d_materns = []
        for dim in range(input_dim):
            one_d_materns.append(
                Matern(input_dim=1, lengthscale=lengthscales[dim], nu=nus[dim])
            )
        self.one_d_materns = one_d_materns
        self.nus = nus
        self.lengthscales = lengthscales

        super().__init__(input_dim=input_dim)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:

        kernel_eval = 1.0

        if x1 is None:
            for dim in range(self.input_dim):
                kernel_eval *= self.one_d_materns[dim]._evaluate(x0[..., dim, None])
        else:
            for dim in range(self.input_dim):
                kernel_eval *= self.one_d_materns[dim]._evaluate(
                    x0[..., dim, None], x1[..., dim, None]
                )

        return kernel_eval
