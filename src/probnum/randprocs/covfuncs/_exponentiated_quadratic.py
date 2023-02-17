"""Exponentiated quadratic covariance function."""

import functools
from typing import Optional

import numpy as np

from probnum.typing import ArrayLike, ShapeLike

from ._covariance_function import CovarianceFunction, IsotropicMixin


class ExpQuad(CovarianceFunction, IsotropicMixin):
    r"""Exponentiated quadratic covariance function.

    Covariance function defined by

    .. math ::
        k(x_0, x_1) = \exp \left( -\frac{\lVert x_0 - x_1 \rVert_2^2}{2 l^2} \right).

    This covariance function is also known as the squared exponential (SE) or radial
    basis function (RBF) kernel.

    Parameters
    ----------
    input_shape
        Shape of the covariance function's input.
    lengthscale
        Lengthscale :math:`l` of the covariance function. Describes the input scale on
        which the process varies.

    See Also
    --------
    RatQuad : Rational quadratic covariance function.
    Matern : Matern covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.covfuncs import ExpQuad
    >>> K = ExpQuad((), lengthscales=0.1)
    >>> xs = np.linspace(0, 1, 3)
    >>> K.matrix(xs)
    array([[1.00000000e+00, 3.72665317e-06, 1.92874985e-22],
           [3.72665317e-06, 1.00000000e+00, 3.72665317e-06],
           [1.92874985e-22, 3.72665317e-06, 1.00000000e+00]])
    """

    def __init__(self, input_shape: ShapeLike, *, lengthscales: ArrayLike = 1.0):
        super().__init__(input_shape=input_shape)

        # Input lengthscales
        lengthscales = np.asarray(
            lengthscales if lengthscales is not None else 1.0, dtype=np.double
        )

        if np.any(lengthscales <= 0):
            raise ValueError(f"Lengthscales l={lengthscales} must be positive.")

        np.broadcast_to(  # Check if the lengthscales broadcast to the input dimension
            lengthscales, self.input_shape
        )

        self._lengthscales = lengthscales

    @property
    def lengthscales(self) -> np.ndarray:
        r"""Input lengthscales along the different input dimensions."""
        return self._lengthscales

    @property
    def lengthscale(self) -> np.ndarray:  # pylint: disable=missing-raises-doc
        """Deprecated."""
        if self._lengthscales.shape == ():
            return self._lengthscales

        raise ValueError("There is more than one lengthscale.")

    @functools.cached_property
    def _scale_factors(self) -> np.ndarray:
        return np.sqrt(0.5) / self._lengthscales

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        if x1 is None:
            return np.ones_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return np.exp(
            -self._squared_euclidean_distances(
                x0, x1, scale_factors=self._scale_factors
            )
        )
