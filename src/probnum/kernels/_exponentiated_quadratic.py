"""Exponentiated quadratic kernel."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class ExpQuad(Kernel[_InputType]):
    """Exponentiated quadratic / RBF kernel.

    Covariance function defined by :math:`k(x_0, x_1) = \\exp(-\\frac{\\lVert x_0 -
    x_1 \\rVert^2}{2l^2})`. This kernel is also known as the squared exponential or
    radial basis function kernel.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    lengthscale
        Lengthscale of the kernel. Describes the input scale on which the process
        varies.

    See Also
    --------
    RatQuad : Rational quadratic kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kerns import ExpQuad
    >>> K = ExpQuad(input_dim=1, lengthscale=0.1)
    >>> K(np.array([[1], [.1], [.5]]))
    array([[1.00000000e+00, 2.57675711e-18, 3.72665317e-06],
           [2.57675711e-18, 1.00000000e+00, 3.35462628e-04],
           [3.72665317e-06, 3.35462628e-04, 1.00000000e+00]])
    """

    def __init__(self, input_dim: IntArgType, lengthscale: ScalarArgType = 1.0):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        super().__init__(input_dim=input_dim, output_dim=1)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        # Check and reshape inputs
        x0, x1, equal_inputs = self._check_and_transform_input(x0, x1)
        x0_originalshape = x0.shape
        x1_originalshape = x1.shape

        # Pre-compute norms with einsum for efficiency
        x0 = np.atleast_2d(x0)
        x0_norm_sq = np.einsum("nd,nd->n", x0, x0)

        if equal_inputs:
            x1 = x0
            x1_norm_sq = x0_norm_sq
        else:
            x1 = np.atleast_2d(x1)
            x1_norm_sq = np.einsum("nd,nd->n", x1, x1)

        # Kernel matrix via broadcasting
        kernmat = np.exp(
            -(x0_norm_sq[:, None] + x1_norm_sq[None, :] - 2 * x0 @ x1.T)
            / (2 * self.lengthscale ** 2)
        )
        return self._transform_kernelmatrix(
            kerneval=kernmat, x0_shape=x0_originalshape, x1_shape=x1_originalshape
        )
