"""Exponentiated quadratic kernel."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = TypeVar("InputType")


class ExpQuad(Kernel[_InputType]):
    """Exponentiated quadratic / RBF kernel.

    Covariance function defined by :math:`k(x_0, x_1) = \\exp(-\\frac{\\lVert x_0 -
    x_1 \\rVert^2}{2l^2})`. This kernel is also known as the squared exponential or
    radial basis function kernel.

    Parameters
    ----------
    lengthscale
        Lengthscale of the kernel. Describes the input scale on which the process
        varies.
    """

    def __init__(self, lengthscale: ScalarArgType):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        super().__init__(kernel=self._kernel, output_dim=1)

    def _kernel(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        # Transform into 2d array
        x0 = _utils.as_colvec(x0)
        # Pre-compute norms with einsum for performance
        x0_norm = np.einsum("ij,ij->i", x0, x0)

        if x1 is None:
            x1 = x0
            x1_norm = x0_norm
        else:
            x1 = _utils.as_colvec(x1)
            x1_norm = np.einsum("ij,ij->i", x1, x1)

        # Kernel matrix
        return np.exp(
            -(x0_norm[:, None] + x1_norm[None, :] - 2 * x0 @ x1.T)
            / (2 * self.lengthscale ** 2)
        )
