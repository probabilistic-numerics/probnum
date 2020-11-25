"""Polynomial kernel."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType

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
    >>> from probnum.kerns import Polynomial
    >>> K = Polynomial(input_dim=2, constant=1.0, exponent=3)
    >>> K(np.array([[1, -1], [-1, 0]]))
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

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        # Check and reshape inputs
        x0, x1, equal_inputs = self._check_and_transform_input(x0, x1)
        x0_originalshape = x0.shape
        x1_originalshape = x1.shape

        # Compute kernel matrix
        x0 = np.atleast_2d(x0)
        x1 = np.atleast_2d(x1)

        kernmat = (x0 @ x1.T + self.constant) ** self.exponent

        return self._transform_kernelmatrix(
            kerneval=kernmat, x0_shape=x0_originalshape, x1_shape=x1_originalshape
        )
