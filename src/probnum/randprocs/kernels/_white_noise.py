"""White noise kernel."""

from typing import Optional

from probnum import backend
from probnum.typing import ArrayType, IntLike, ScalarLike

from ._kernel import Kernel


class WhiteNoise(Kernel):
    r"""White noise kernel.

    Kernel representing independent and identically distributed white noise

    .. math ::
        k(x_0, x_1) = \sigma^2 \delta(x_0, x_1).

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    sigma :
        Noise level :math:`\sigma`.
    """

    def __init__(self, input_dim: IntLike, sigma: ScalarLike = 1.0):
        self.sigma = backend.as_scalar(sigma)
        self._sigma_sq = self.sigma ** 2
        super().__init__(input_dim=input_dim)

    def _evaluate(self, x0: ArrayType, x1: Optional[ArrayType]) -> ArrayType:
        if x1 is None:
            return backend.full_like(x0[..., 0], self._sigma_sq)

        return self._sigma_sq * backend.all(x0 == x1, axis=-1)
