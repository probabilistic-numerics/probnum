"""Rational quadratic covariance function."""

from typing import Optional

import numpy as np

from probnum.typing import ScalarLike, ShapeLike
import probnum.utils as _utils

from ._covariance_function import CovarianceFunction, IsotropicMixin


class RatQuad(CovarianceFunction, IsotropicMixin):
    r"""Rational quadratic covariance function.

    Covariance function defined by

    .. math::
        :nowrap:

        \begin{equation}
            k(x_0, x_1)
            = \left(
                1 + \frac{\lVert x_0 - x_1 \rVert_2^2}{2 \alpha l^2}
            \right)^{-\alpha},
        \end{equation}

    where :math:`\alpha > 0`. For :math:`\alpha \rightarrow \infty` the rational
    quadratic covariance function converges to the :class:`~probnum.randprocs.covfuncs.\
    ExpQuad` covariance function.

    Parameters
    ----------
    input_shape
        Shape of the covariance function's input.
    lengthscale
        Lengthscale :math:`l` of the covariance function. Describes the input scale on
        which the process varies.
    alpha
        Scale mixture :math:`\alpha`. Positive constant determining the weighting
        between different lengthscales.

    See Also
    --------
    ExpQuad : Exponentiated Quadratic / RBF covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.covfuncs import RatQuad
    >>> K = RatQuad(input_shape=1, lengthscale=0.1, alpha=3)
    >>> xs = np.linspace(0, 1, 3)[:, None]
    >>> K(xs[:, None, :], xs[None, :, :])
    array([[1.00000000e+00, 7.25051190e-03, 1.81357765e-04],
           [7.25051190e-03, 1.00000000e+00, 7.25051190e-03],
           [1.81357765e-04, 7.25051190e-03, 1.00000000e+00]])
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscale: ScalarLike = 1.0,
        alpha: ScalarLike = 1.0,
    ):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        self.alpha = _utils.as_numpy_scalar(alpha)
        if not self.alpha > 0:
            raise ValueError(f"Scale mixture alpha={self.alpha} must be positive.")
        super().__init__(input_shape=input_shape)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        if x1 is None:
            return np.ones_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return (
            1.0
            + (
                self._squared_euclidean_distances(x0, x1)
                / (2.0 * self.alpha * self.lengthscale**2)
            )
        ) ** -self.alpha
