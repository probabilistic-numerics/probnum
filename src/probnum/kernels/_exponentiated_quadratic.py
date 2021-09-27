"""Exponentiated quadratic kernel."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class ExpQuad(Kernel[_InputType]):
    """Exponentiated quadratic / RBF kernel.

    Covariance function defined by :math:`k(x_0, x_1) = \\exp \\big(-\\frac{\\lVert
    x_0 - x_1 \\rVert^2}{2l^2}\\big)`. This kernel is also known as the squared
    exponential or radial basis function kernel.

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
    Matern : Matern kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import ExpQuad
    >>> K = ExpQuad(input_dim=1, lengthscale=0.1)
    >>> xs = np.linspace(0, 1, 3)[:, None]
    >>> K(xs[:, None, :], xs[None, :, :])
    array([[1.00000000e+00, 3.72665317e-06, 1.92874985e-22],
           [3.72665317e-06, 1.00000000e+00, 3.72665317e-06],
           [1.92874985e-22, 3.72665317e-06, 1.00000000e+00]])
    """

    def __init__(self, input_dim: IntArgType, lengthscale: ScalarArgType = 1.0):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        super().__init__(input_dim=input_dim, output_dim=1)

    def _evaluate(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        if x1 is None:
            return np.ones_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[:-1] + (1, 1),
            )

        sqdists = np.sum((x0 - x1) ** 2, axis=-1)
        kernmat = np.exp(-1.0 / (2.0 * self.lengthscale ** 2) * sqdists)

        return kernmat[..., None, None]
