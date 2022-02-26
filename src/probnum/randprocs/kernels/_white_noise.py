"""White noise kernel."""

from typing import Optional

from probnum import backend
from probnum.typing import ScalarLike, ShapeLike

from ._kernel import Kernel


class WhiteNoise(Kernel):
    r"""White noise kernel.

    Kernel representing independent and identically distributed white noise

    .. math ::
        k(x_0, x_1) = \sigma^2 \delta(x_0, x_1).

    Parameters
    ----------
    input_shape :
        Shape of the kernel's input.
    sigma :
        Noise level :math:`\sigma`.
    """

    def __init__(self, input_shape: ShapeLike, sigma: ScalarLike = 1.0):
        self.sigma = backend.as_scalar(sigma)
        self._sigma_sq = self.sigma**2
        super().__init__(input_shape=input_shape)

    def _evaluate(
        self, x0: backend.ndarray, x1: Optional[backend.ndarray]
    ) -> backend.ndarray:
        if x1 is None:
            return backend.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._sigma_sq,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        if self.input_shape == ():
            return self._sigma_sq * (x0 == x1)

        return self._sigma_sq * backend.all(x0 == x1, axis=-1)
