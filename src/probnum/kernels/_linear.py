"""Linear covariance function."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class Linear(Kernel[_InputType]):
    """Linear kernel.

    Linear covariance function defined by :math:`k(x_0, x_1) = x_0^\\top x_1 + c`.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    constant
        Constant offset :math:`c`.

    See Also
    --------
    Polynomial : Polynomial covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import Linear
    >>> K = Linear(input_dim=2)
    >>> xs = np.array([[1, 2], [2, 3]])
    >>> K.matrix(xs)
    array([[ 5.,  8.],
           [ 8., 13.]])
    """

    def __init__(self, input_dim: IntArgType, constant: ScalarArgType = 0.0):
        self.constant = _utils.as_numpy_scalar(constant)
        super().__init__(input_dim=input_dim, output_dim=1)

    def _evaluate(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        kernmat = self._euclidean_inner_products(x0, x1) + self.constant

        return kernmat[..., None, None]
