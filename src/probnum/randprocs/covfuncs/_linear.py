"""Linear covariance function."""

from __future__ import annotations

from typing import Optional

import numpy as np

from probnum.typing import ScalarLike, ShapeLike
import probnum.utils as _utils

from ._covariance_function import CovarianceFunction


class Linear(CovarianceFunction):
    r"""Linear covariance function.

    Linear covariance function defined by

    .. math ::
        k(x_0, x_1) = x_0^\top x_1 + c.

    Parameters
    ----------
    input_shape
        Shape of the covariance function's input.
    constant
        Constant offset :math:`c`.

    See Also
    --------
    Polynomial : Polynomial covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.covfuncs import Linear
    >>> K = Linear(input_shape=2)
    >>> xs = np.array([[1, 2], [2, 3]])
    >>> K.matrix(xs)
    array([[ 5.,  8.],
           [ 8., 13.]])
    """

    def __init__(self, input_shape: ShapeLike, constant: ScalarLike = 0.0):
        self.constant = _utils.as_numpy_scalar(constant)
        super().__init__(input_shape=input_shape)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._euclidean_inner_products(x0, x1) + self.constant
