"""Linear covariance function."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = TypeVar("InputType")


class Linear(Kernel[_InputType]):
    """Linear kernel.

    Covariance function defined by :math:`k(x_0, x_1) = (x_0 - c)^\\top(x_1 - c)`.

    Parameters
    ----------
    constant
        Constant shift.
    """

    def __init__(self, constant: ScalarArgType):
        self.constant = _utils.as_numpy_scalar(constant)
        super().__init__(fun=self._fun, output_dim=1)

    def _fun(self, x0: _InputType, x1: _InputType):
        return np.inner(x0 - self.constant, x1 - self.constant)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        if x1 is None:
            x1 = x0

        x0 = _utils.as_colvec(x0)
        x1 = _utils.as_colvec(x1)

        return np.sum(
            np.multiply.outer(x0 - self.constant, x1 - self.constant), axis=(1, 3)
        )
