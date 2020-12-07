"""Exponentiated quadratic kernel."""

from typing import Optional

import numpy as np
import scipy.spatial.distance

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType

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

        # Compute pairwise euclidean distances ||x0 - x1|| / l
        x0 = np.atleast_2d(x0)
        if equal_inputs:
            pdists = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(x0 / self.lengthscale, metric="euclidean")
            )
        else:
            x1 = np.atleast_2d(x1)
            pdists = scipy.spatial.distance.cdist(
                x0 / self.lengthscale, x1 / self.lengthscale, metric="euclidean"
            )

        # Kernel matrix
        kernmat = np.exp(-(pdists ** 2) / 2.0)
        return self._transform_kernelmatrix(
            kerneval=kernmat, x0_shape=x0_originalshape, x1_shape=x1_originalshape
        )
