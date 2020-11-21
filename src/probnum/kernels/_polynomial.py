"""Polynomial kernel."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = TypeVar("InputType")
# TODO: fix correct types for kernels here


class Polynomial(Kernel[_InputType]):
    """Polynomial kernel.

    Covariance function defined by :math:`k(x_0, x_1) = (x_0^\\top x_1 + c)^q`.

    Parameters
    ----------
    constant
        Constant offset :math:`c`.
    exponent
        Exponent :math:`q` of the polynomial.
    """

    def __init__(self, constant: ScalarArgType = 0.0, exponent: ScalarArgType = 1.0):
        self.constant = _utils.as_numpy_scalar(constant)
        self.exponent = _utils.as_numpy_scalar(exponent)
        super().__init__(kernel=self._kernel, output_dim=1)

    def _kernel(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        x0 = np.atleast_2d(x0)
        if x1 is None:
            x1 = x0
        else:
            x1 = np.atleast_2d(x1)

        return (x0 @ x1.T + self.constant) ** self.exponent
