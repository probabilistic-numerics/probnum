"""White noise kernel."""

from typing import Optional

import numpy as np

from probnum.typing import ScalarLike, ShapeLike

from ._kernel import Kernel


class WhiteNoise(Kernel):
    r"""White noise kernel.

    Kernel representing independent and identically distributed white noise

    .. math ::
        k(x_0, x_1) = \sigma^2 \delta(x_0, x_1).

    Parameters
    ----------
    input_shape
        Shape of the kernel's input.
    sigma_sq
        Positive noise level :math:`\sigma^2 > 0`.
    """

    def __init__(self, input_shape: ShapeLike, sigma_sq: ScalarLike = 1.0):

        super().__init__(input_shape=input_shape, sigma_sq=sigma_sq)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self.sigma_sq,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        if self.input_shape == ():
            return self.sigma_sq * (x0 == x1)

        return self.sigma_sq * np.all(x0 == x1, axis=-1)
