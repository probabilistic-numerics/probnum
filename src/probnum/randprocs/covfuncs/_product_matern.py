"""Tensor product of 1D Matérn covariance functions."""

from typing import Optional, Union

import numpy as np

from probnum import utils as _utils
from probnum.typing import ScalarLike, ShapeLike

from ._covariance_function import CovarianceFunction
from ._matern import Matern


class ProductMatern(CovarianceFunction):
    r"""Tensor product of one-dimensional Matérn covariance functions.

    Covariance function defined as a product of one-dimensional Matérn covariance
    functions: :math:`k(x_0, x_1) = \prod_{i=1}^d k_i(x_{0,i}, x_{1,i})`,
    where :math:`x_0 = (x_{0,i}, \ldots, x_{0,d})` and :math:`x_0 = (x_{0,i}, \ldots,
    x_{0,d})` and :math:`k_i` are one-dimensional Matérn covariance functions.

    Parameters
    ----------
    input_shape
        Shape of the covariance function's input.
    lengthscales
        Lengthscales of the one-dimensional Matérn covariance functions. Describes the
        input scale on which the process varies. If a scalar, the same lengthscale is
        used in each dimension.
    nus
        Hyperparameters controlling differentiability of the one-dimensional Matérn
        covariance functions. If a scalar, the same smoothness is used in each
        dimension.

    See Also
    --------
    Matern : Stationary Matérn covariance function.
    ExpQuad : Exponentiated Quadratic covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.covfuncs import ProductMatern
    >>> lengthscales = np.array([0.1, 1.2])
    >>> nus = np.array([0.5, 3.5])
    >>> K = ProductMatern(input_shape=(2,), lengthscales=lengthscales, nus=nus)
    >>> xs = np.array([[0.0, 0.5], [1.0, 1.0], [0.5, 0.2]])
    >>> K.matrix(xs)
    array([[1.00000000e+00, 4.03712525e-05, 6.45332482e-03],
           [4.03712525e-05, 1.00000000e+00, 5.05119251e-03],
           [6.45332482e-03, 5.05119251e-03, 1.00000000e+00]])

    Raises
    ------
    ValueError
        If covariance function input is scalar, but  ``lengthscales`` or ``nus`` are
        not.
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscales: Union[np.ndarray, ScalarLike],
        nus: Union[np.ndarray, ScalarLike],
    ):
        input_shape = _utils.as_shape(input_shape)
        if input_shape == () and not (np.isscalar(lengthscales) and np.isscalar(nus)):
            raise ValueError(
                f"'lengthscales' and 'nus' must be scalar if 'input_shape' is "
                f"{input_shape}."
            )

        input_dim = 1 if input_shape == () else input_shape[0]

        # If only single scalar lengthcsale or nu is given, use this in every dimension
        def _expand_array(x, ndim):
            return np.full((ndim,), _utils.as_numpy_scalar(x))

        if isinstance(lengthscales, np.ndarray):
            if lengthscales.shape == ():
                lengthscales = _expand_array(lengthscales, input_dim)
        if isinstance(nus, np.ndarray):
            if nus.shape == ():
                nus = _expand_array(nus, input_dim)

        # also expand if scalars are given
        if np.isscalar(lengthscales):
            lengthscales = _expand_array(lengthscales, input_dim)
        if np.isscalar(nus):
            nus = _expand_array(nus, input_dim)

        univariate_materns = []
        for dim in range(input_dim):
            univariate_materns.append(
                Matern(input_shape=(), lengthscales=lengthscales[dim], nu=nus[dim])
            )
        self.univariate_materns = univariate_materns
        self.nus = nus
        self.lengthscales = lengthscales

        super().__init__(input_shape=input_shape)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:

        # scalar case is same as a scalar Matern
        if self.input_shape == ():
            if x1 is None:
                return self.univariate_materns[0](x0, None)
            return self.univariate_materns[0](x0, x1)

        # product case
        (input_dim,) = self.input_shape

        k_x0_x1 = 1.0
        if x1 is None:
            for dim in range(input_dim):
                k_x0_x1 *= self.univariate_materns[dim](x0[..., dim], None)
        else:
            for dim in range(input_dim):
                k_x0_x1 *= self.univariate_materns[dim](x0[..., dim], x1[..., dim])

        return k_x0_x1
