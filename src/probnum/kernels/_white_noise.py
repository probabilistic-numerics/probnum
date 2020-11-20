"""White noise kernel."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = TypeVar("InputType")


class WhiteNoise(Kernel[_InputType]):
    """White noise kernel.

    Kernel representing independent and identically distributed white noise :math:`k(
    x_0, x_1) = \\sigma^2 \\delta(x_0, x_1)`.

    Parameters
    ----------
    sigma
        Noise level.
    """

    def __init__(self, sigma: ScalarArgType):
        self.sigma = _utils.as_numpy_scalar(sigma)
        super().__init__(fun=self._fun, output_dim=1)

    def _fun(self, x0: _InputType, x1: _InputType):
        if x0 == x1:
            return self.sigma
        else:
            return 0

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        if x1 is None:
            return self.sigma ** 2 * np.eye(x0.shape[0])

        x0 = _utils.as_colvec(x0)
        x1 = _utils.as_colvec(x1)

        return self.sigma ** 2 * np.equal.outer(x0, x1).squeeze().astype(float)
