"""Polynomial covariance function."""

from typing import Optional

import numpy as np

from probnum.typing import IntLike, ScalarLike, ShapeLike
import probnum.utils as _utils

from ._covariance_function import CovarianceFunction


class Polynomial(CovarianceFunction):
    r"""Polynomial covariance function.

    Covariance function defined by

    .. math ::
        k(x_0, x_1) = (x_0^\top x_1 + c)^q.

    Parameters
    ----------
    input_shape
        Shape of the covariance function's input.
    constant
        Constant offset :math:`c`.
    exponent
        Exponent :math:`q` of the polynomial.

    See Also
    --------
    Linear : Linear covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.covfuncs import Polynomial
    >>> K = Polynomial(input_shape=2, constant=1.0, exponent=3)
    >>> xs = np.array([[1, -1], [-1, 0]])
    >>> K.matrix(xs)
    array([[27.,  0.],
           [ 0.,  8.]])
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        constant: ScalarLike = 0.0,
        exponent: IntLike = 1.0,
    ):
        self.constant = _utils.as_numpy_scalar(constant)
        self.exponent = _utils.as_numpy_scalar(exponent)
        super().__init__(input_shape=input_shape)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        return (self._euclidean_inner_products(x0, x1) + self.constant) ** self.exponent
