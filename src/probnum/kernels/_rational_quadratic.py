"""Rational quadratic kernel."""

from typing import Optional, TypeVar

import numpy as np

import probnum.utils as _utils
from probnum.type import ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class RatQuad(Kernel[_InputType]):
    """Rational quadratic kernel.

    Covariance function defined by :math:`k(x_0, x_1) = (1 + \\frac{\\lVert x_0 -
    x_1 \\rVert^2}{2\\alpha l^2})^{-\\alpha}`, where :math:`\\alpha > 0`. For
    :math:`\\alpha \\rightarrow \\infty` the rational quadratic kernel converges to the
    :class:`~probnum.kernels.ExpQuad` kernel.

    Parameters
    ----------
    lengthscale :
        Lengthscale of the kernel. Describes the input scale on which the process
        varies.
    alpha :
        Scale mixture. Positive constant determining the weighting between different
        lengthscales.
    """

    def __init__(self, lengthscale: ScalarArgType = 1.0, alpha: ScalarArgType = 1.0):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        self.alpha = _utils.as_numpy_scalar(alpha)
        if not self.alpha > 0:
            raise ValueError(f"Scale mixture alpha={self.alpha} must be positive.")
        super().__init__(kernel=self.__call__, output_dim=1)

    def __call__(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:

        # Pre-compute norms with einsum for efficiency
        x0 = np.atleast_2d(x0)
        x0_norm_sq = np.einsum("nd,nd->n", x0, x0)

        if x1 is None:
            x1 = x0
            x1_norm_sq = x0_norm_sq
        else:
            x1 = np.atleast_2d(x1)
            x1_norm_sq = np.einsum("nd,nd->n", x1, x1)

        # Kernel matrix via broadcasting
        return (
            1.0
            + (x0_norm_sq[:, None] + x1_norm_sq[None, :] - 2 * x0 @ x1.T)
            / (self.alpha * self.lengthscale ** 2)
        ) ** (-self.alpha)
