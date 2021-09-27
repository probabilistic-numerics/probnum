"""Polynomial kernel."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class Polynomial(Kernel[_InputType]):
    """Polynomial kernel.

    Covariance function defined by :math:`k(x_0, x_1) = (x_0^\\top x_1 + c)^q`.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
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
    >>> from probnum.kernels import Polynomial
    >>> K = Polynomial(input_dim=2, constant=1.0, exponent=3)
    >>> xs = np.array([[1, -1], [-1, 0]])
    >>> K(xs[:, None, :], xs[None, :, :])
    array([[27.,  0.],
           [ 0.,  8.]])
    """

    def __init__(
        self,
        input_dim: IntArgType,
        constant: ScalarArgType = 0.0,
        exponent: IntArgType = 1.0,
    ):
        self.constant = _utils.as_numpy_scalar(constant)
        self.exponent = _utils.as_numpy_scalar(exponent)
        super().__init__(input_dim=input_dim, output_dim=1)

    def _evaluate(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        kernmat = self._euclidean_inner_products(x0, x1) + self.constant
        kernmat = kernmat ** self.exponent

        return kernmat[..., None, None]
