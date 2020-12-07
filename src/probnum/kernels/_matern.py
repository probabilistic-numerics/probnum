"""Matern kernel."""

from typing import Optional

import numpy as np
import scipy.special

import probnum.utils as _utils
from probnum.type import IntArgType, ScalarArgType

from ._kernel import Kernel

_InputType = np.ndarray


class Matern(Kernel[_InputType]):
    """Matern kernel.

    Covariance function defined by :math:`k(x_0, x_1) = \\frac{1}{\\Gamma(\\nu)2^{
    \\nu-1}}\\Bigg(\\frac{\\sqrt{2\\nu}}{l} \\lVert x_0 , x_1\\rVert \\Bigg)^\\nu
    K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)`

    The Matern
    kernel generalizes the :class:`~probnum.kernels.ExponentiatedQuadratic` kernel
    via its additional parameter :math:`\\nu` controlling the smoothness of the
    function. For :math:`\\nu \\rightarrow \\infty` the Matern kernel converges to
    the :class:`~probnum.kernels.ExponentiatedQuadratic` kernel. A Gaussian process
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
    >>> K(np.linspace(0, 1, 5))
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

        pdists = (
            np.sqrt(x0_norm_sq[:, None] + x1_norm_sq[None, :] - 2 * x0 @ x1.T)
            / self.lengthscale
        )

        if self.nu == 0.5:
            kernmat = np.exp(-pdists)
        elif self.nu == 1.5:
            scaled_pdists = np.sqrt(3) * pdists
            kernmat = (1.0 + scaled_pdists) * np.exp(-scaled_pdists)
        elif self.nu == 2.5:
            scaled_pdists = np.sqrt(5) * pdists
            kernmat = (1.0 + scaled_pdists + scaled_pdists ** 2 / 3.0) * np.exp(
                -scaled_pdists
            )
        elif self.nu == np.inf:
            kernmat = np.exp(-(pdists ** 2) / 2.0)
        else:
            pdists[pdists == 0.0] += np.finfo(float).eps
            scaled_pdists = np.sqrt(2 * self.nu) * pdists
            kernmat = (
                2 ** (1.0 - self.nu)
                / scipy.special.gamma(self.nu)
                * scaled_pdists ** self.nu
                * scipy.special.kv(self.nu, scaled_pdists)
            )

        return self._transform_kernelmatrix(
            kerneval=kernmat, x0_shape=x0_originalshape, x1_shape=x1_originalshape
        )
