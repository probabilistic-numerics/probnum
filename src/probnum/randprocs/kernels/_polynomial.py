"""Polynomial kernel."""

from typing import Optional

import numpy as np

from probnum.typing import IntLike, ScalarLike, ShapeLike
import probnum.utils as _utils

from ._kernel import Kernel


class Polynomial(Kernel):
    r"""Polynomial kernel.

    Covariance function defined by

    .. math ::
        k(x_0, x_1) = \sigma^2 (x_0^\top x_1 + c)^q.

    Parameters
    ----------
    input_shape
        Shape of the kernel's input.
    sigma_sq
        Positive kernel output scaling parameter :math:`\sigma^2 \geq 0`.
    constant
        Constant offset :math:`c`.
    exponent
        Exponent :math:`q` of the polynomial.

    See Also
    --------
    Linear : Linear covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.kernels import Polynomial
    >>> K = Polynomial(input_shape=2, constant=1.0, exponent=3)
    >>> xs = np.array([[1, -1], [-1, 0]])
    >>> K.matrix(xs)
    array([[27.,  0.],
           [ 0.,  8.]])
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        sigma_sq: ScalarLike = 1.0,
        constant: ScalarLike = 0.0,
        exponent: IntLike = 1.0,
    ):
        self.constant = _utils.as_numpy_scalar(constant)
        self.exponent = _utils.as_numpy_scalar(exponent)
        super().__init__(input_shape=input_shape, sigma_sq=sigma_sq)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        return (
            self.sigma_sq
            * (self._euclidean_inner_products(x0, x1) + self.constant) ** self.exponent
        )
