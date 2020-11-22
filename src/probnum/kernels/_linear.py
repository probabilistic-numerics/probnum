"""Linear covariance function."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class Linear(Kernel[_InputType]):
    """Linear kernel.

    Linear covariance function defined by :math:`k(x_0, x_1) = (x_0 - c)^\\top(x_1 -
    c)`.

    Parameters
    ----------
    shift :
        Constant shift :math:`c`.

    See Also
    --------
    Polynomial : Polynomial covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import Linear
    >>> K = Linear()
    >>> K(np.array([[1, 2], [2, 3]]))
    array([[5., 8.],
           [8., 13.]])
    """

    def __init__(self, shift: ScalarArgType = 0.0):
        self.shift = _utils.as_numpy_scalar(shift)
        super().__init__(kernel=self.__call__, output_dim=1)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        x0 = np.atleast_2d(x0)
        if x1 is None:
            x1 = x0
        else:
            x1 = np.atleast_2d(x1)

        return (x0 - self.shift) @ (x1 - self.shift).T
