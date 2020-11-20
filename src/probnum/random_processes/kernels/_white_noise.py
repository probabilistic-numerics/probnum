"""White noise covariance function."""

from typing import Optional, TypeVar

import numpy as np

from ._kernel import Kernel

_InputType = TypeVar("InputType")


class Constant(Kernel(_InputType)):
    """White noise covariance function.

    Kernel representing independent and identically distributed white noise :math:`k(
    x_0, x_1) = \\sigma^2 \\vardelta_{0, 1}`.

    Parameters
    ----------
    sigma
        Noise level.
    """

    def __init__(self, sigma: np.float_):
        self.sigma = sigma
        super().__init__(fun=lambda x0, x1: sigma if x0 == x1 else 0)

    def __call__(self, x0: [_InputType], x1: Optional[_InputType] = None) -> np.ndarray:
        raise NotImplementedError
