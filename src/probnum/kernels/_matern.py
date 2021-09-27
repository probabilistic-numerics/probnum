"""Matern kernel."""

from typing import Optional

import numpy as np
import scipy.spatial.distance
import scipy.special

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class Matern(Kernel[_InputType]):
    """Matern kernel.

    Covariance function defined by :math:`k(x_0, x_1) = \\frac{1}{\\Gamma(\\nu)2^{
    \\nu-1}}\\big(\\frac{\\sqrt{2\\nu}}{l} \\lVert x_0 , x_1\\rVert \\big)^\\nu
    K_\\nu\\big(\\frac{\\sqrt{2\\nu}}{l} \\lVert x_0 , x_1 \\rVert \\big)`, where
    :math:`K_\\nu` is a modified Bessel function. The Matern
    kernel generalizes the :class:`~probnum.kernels.ExpQuad` kernel
    via its additional parameter :math:`\\nu` controlling the smoothness of the
    function. For :math:`\\nu \\rightarrow \\infty` the Matern kernel converges to
    the :class:`~probnum.kernels.ExpQuad` kernel. A Gaussian process
    with Matern covariance function is :math:`\\lceil \\nu \\rceil - 1` times
    differentiable.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    lengthscale :
        Lengthscale of the kernel. Describes the input scale on which the process
        varies.
    nu :
        Hyperparameter controlling differentiability.

    See Also
    --------
    ExpQuad : Exponentiated Quadratic / RBF kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import Matern
    >>> K = Matern(input_dim=1, lengthscale=0.1, nu=2.5)
    >>> xs = np.linspace(0, 1, 3)[:, None]
    >>> K(xs[:, None, :], xs[None, :, :])
    array([[1.00000000e+00, 7.50933789e-04, 3.69569622e-08],
           [7.50933789e-04, 1.00000000e+00, 7.50933789e-04],
           [3.69569622e-08, 7.50933789e-04, 1.00000000e+00]])
    """

    def __init__(
        self,
        input_dim: IntArgType,
        lengthscale: ScalarArgType = 1.0,
        nu: ScalarArgType = 1.5,
    ):
        # pylint: disable="invalid-name"
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        if not self.lengthscale > 0:
            raise ValueError(f"Lengthscale l={self.lengthscale} must be positive.")
        self.nu = _utils.as_numpy_scalar(nu)
        if not self.nu > 0:
            raise ValueError(f"Hyperparameter nu={self.nu} must be positive.")

        super().__init__(input_dim=input_dim, output_dim=1)

    def _evaluate(self, x0: _InputType, x1: Optional[_InputType] = None) -> np.ndarray:
        if x1 is None:
            dists = np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[:-1],
            )
        else:
            dists = np.linalg.norm(x0 - x1, ord=2, axis=-1)

        # Kernel matrix computation dependent on differentiability
        if self.nu == 0.5:
            kernmat = np.exp(-1.0 / self.lengthscale * dists)
        elif self.nu == 1.5:
            scaled_dists = -np.sqrt(3) / self.lengthscale * dists
            kernmat = (1.0 + scaled_dists) * np.exp(-scaled_dists)
        elif self.nu == 2.5:
            scaled_dists = np.sqrt(5) / self.lengthscale * dists
            kernmat = (1.0 + scaled_dists + scaled_dists ** 2 / 3.0) * np.exp(
                -scaled_dists
            )
        elif self.nu == np.inf:
            kernmat = np.exp(-1.0 / (2.0 * self.lengthscale ** 2) * dists ** 2)
        else:
            # The modified Bessel function K_nu is not defined for z=0
            dists = np.maximum(dists, np.finfo(dists.dtype).eps)

            scaled_dists = np.sqrt(2 * self.nu) / self.lengthscale * dists
            kernmat = (
                2 ** (1.0 - self.nu)
                / scipy.special.gamma(self.nu)
                * scaled_dists ** self.nu
                * scipy.special.kv(self.nu, scaled_dists)
            )

        return kernmat[..., None, None]
