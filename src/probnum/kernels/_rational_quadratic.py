"""Rational quadratic kernel."""

from typing import Optional

import numpy as np
import scipy.spatial.distance

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class RatQuad(Kernel[_InputType]):
    """Rational quadratic kernel.

    Covariance function defined by :math:`k(x_0, x_1) = \\big(1 + \\frac{\\lVert x_0 -
    x_1 \\rVert^2}{2\\alpha l^2}\\big)^{-\\alpha}`, where :math:`\\alpha > 0`. For
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
    >>> from probnum.kernels import RatQuad
    >>> K = RatQuad(input_dim=1, lengthscale=0.1, alpha=3)
    >>> K(np.linspace(0, 1, 3)[:, None])
    array([[1.00000000e+00, 7.25051190e-03, 1.81357765e-04],
           [7.25051190e-03, 1.00000000e+00, 7.25051190e-03],
           [1.81357765e-04, 7.25051190e-03, 1.00000000e+00]])
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

        # Kernel matrix
        kernmat = (1.0 + pdists ** 2 / (2.0 * self.alpha)) ** (-self.alpha)

        return Kernel._reshape_kernelmatrix(kernmat, newshape=kernshape)
