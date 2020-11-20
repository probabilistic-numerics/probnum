"""Exponentiated quadratic kernel."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = TypeVar("InputType")


class ExpQuad(Kernel[_InputType]):
    """Exponentiated quadratic kernel.

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
        super().__init__(fun=self._fun, output_dim=1)

    def _fun(self, x0: _InputType, x1: _InputType):
        return np.exp(-np.linalg.norm(x0 - x1) ** 2 / (2 * self.lengthscale ** 2))

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        # Transform into 2d arrays
        x0 = _utils.as_colvec(x0)
        x1 = _utils.as_colvec(x1)

        # Pre-compute norms with einsum for performance
        x0_norm = np.einsum("ij,ij->i", x0, x0)
        x1_norm = np.einsum("ij,ij->i", x1, x1)

        # Kernel matrix
        return np.exp(
            -(x0_norm[:, None] + x1_norm[None, :] - 2 * np.dot(x0, x1.T))
            / (2 * self.lengthscale ** 2)
        )
