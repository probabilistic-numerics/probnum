"""White noise kernel."""

from typing import Optional

from probnum import backend
from probnum.typing import ArrayType, ScalarLike, ShapeLike

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
        Noise level :math:`\sigma^2 \geq 0`.
    """

    def __init__(self, input_shape: ShapeLike, sigma_sq: ScalarLike = 1.0):

        if sigma_sq < 0:
            raise ValueError(f"Noise level sigma_sq={sigma_sq} must be non-negative.")

        self.sigma_sq = backend.asscalar(sigma_sq)

        super().__init__(input_shape=input_shape)

    def _evaluate(self, x0: ArrayType, x1: Optional[ArrayType]) -> ArrayType:
        if x1 is None:
            return backend.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self.sigma_sq,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        if self.input_shape == ():
            return self.sigma_sq * (x0 == x1)

        return self.sigma_sq * backend.all(x0 == x1, axis=-1)
