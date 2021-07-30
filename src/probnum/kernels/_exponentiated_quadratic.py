"""Exponentiated quadratic kernel."""

from typing import Optional

import numpy as np
import scipy.spatial.distance

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
    >>> K(np.linspace(0, 1, 3)[:, None])
    array([[1.00000000e+00, 3.72665317e-06, 1.92874985e-22],
           [3.72665317e-06, 1.00000000e+00, 3.72665317e-06],
           [1.92874985e-22, 3.72665317e-06, 1.00000000e+00]])
    """

    def __init__(self, input_dim: IntArgType, lengthscale: ScalarArgType = 1.0):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        super().__init__(input_dim=input_dim, output_dim=1)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:

        x0, x1, kernshape = self._check_and_reshape_inputs(x0, x1)

        # Compute pairwise euclidean distances ||x0 - x1|| / l
        if x1 is None:
            pdists = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(x0 / self.lengthscale, metric="euclidean")
            )
        else:
            pdists = scipy.spatial.distance.cdist(
                x0 / self.lengthscale, x1 / self.lengthscale, metric="euclidean"
            )

        # Compute kernel matrix
        kernmat = np.exp(-(pdists ** 2) / 2.0)

        return Kernel._reshape_kernelmatrix(kernmat, newshape=kernshape)
