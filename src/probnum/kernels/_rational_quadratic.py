"""Rational quadratic kernel."""

from typing import Optional

import numpy as np

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType

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
    input_dim :
        Input dimension of the kernel.
    lengthscale :
        Lengthscale of the kernel. Describes the input scale on which the process
        varies.
    alpha :
        Scale mixture. Positive constant determining the weighting between different
        lengthscales.

    See Also
    --------
    ExpQuad : Exponentiated Quadratic / RBF kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kerns import RatQuad
    >>> K = RatQuad(input_dim=1, lengthscale=0.1, alpha=3)
    >>> K(np.array([[1], [.1], [.5]]))
    array([[1.00000000e+00, 4.55539359e-05, 1.22995627e-03],
           [4.55539359e-05, 1.00000000e+00, 3.93643388e-03],
           [1.22995627e-03, 3.93643388e-03, 1.00000000e+00]])
    """

    def __init__(
        self,
        input_dim: IntArgType,
        lengthscale: ScalarArgType = 1.0,
        alpha: ScalarArgType = 1.0,
    ):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        self.alpha = _utils.as_numpy_scalar(alpha)
        if not self.alpha > 0:
            raise ValueError(f"Scale mixture alpha={self.alpha} must be positive.")
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
        kernmat = (
            1.0
            + (x0_norm_sq[:, None] + x1_norm_sq[None, :] - 2 * x0 @ x1.T)
            / (self.alpha * self.lengthscale ** 2)
        ) ** (-self.alpha)

        return self._transform_kernelmatrix(
            kerneval=kernmat, x0_shape=x0_originalshape, x1_shape=x1_originalshape
        )
