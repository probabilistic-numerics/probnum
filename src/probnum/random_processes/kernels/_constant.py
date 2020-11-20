"""Constant valued covariance function."""

from typing import Optional, TypeVar

import numpy as np

from ._kernel import Kernel

_InputType = TypeVar("InputType")


class Constant(Kernel(_InputType)):
    """Constant-valued covariance function.

    Defines a kernel evaluating to a constant :math:`k(x_0, x_1) = c`.

    Parameters
    ----------
    constant
        Constant value of the kernel.
    """

    def __init__(self, constant: np.float_):
        self.constant = constant
        super().__init__(fun=lambda x0, x1: constant)

    def __call__(self, x0: [_InputType], x1: Optional[_InputType] = None) -> np.ndarray:
        n0 = x0.shape[0]
        if x1 is None:
            n1 = n0
        else:
            n1 = x1.shape[0]

        return self.constant * np.ones(shape=(n0, n1))
