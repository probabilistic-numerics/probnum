"""Linear covariance function."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class Linear(Kernel[_InputType]):
    """Linear kernel.

    Linear covariance function defined by :math:`k(x_0, x_1) = x_0^\\top x_1 + c`.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    constant
        Constant offset :math:`c`.

    See Also
    --------
    Polynomial : Polynomial covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import Linear
    >>> K = Linear(input_dim=2)
    >>> K(np.array([[1, 2], [2, 3]]))
    array([[ 5.,  8.],
           [ 8., 13.]])
    """

    def __init__(self, input_dim: IntArgType, constant: ScalarArgType = 0.0):
        self.constant = _utils.as_numpy_scalar(constant)
        super().__init__(input_dim=input_dim, output_dim=1)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:

        # Check and reshape inputs
        x0, x1, equal_inputs = self._check_and_transform_input(x0, x1)
        x0_originalshape = x0.shape
        x1_originalshape = x1.shape

        # Compute kernel matrix
        x0 = np.atleast_2d(x0)
        x1 = np.atleast_2d(x1)

        kernmat = x0 @ x1.T + self.constant

        return self._transform_kernelmatrix(
            kerneval=kernmat, x0_shape=x0_originalshape, x1_shape=x1_originalshape
        )
