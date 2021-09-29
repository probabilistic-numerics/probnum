"""Matern kernel."""

from typing import Optional

import numpy as np
import scipy.spatial.distance
import scipy.special

import probnum.utils as _utils
from probnum.typing import IntArgType, ScalarArgType

from ._kernel import IsotropicMixin, Kernel


class Matern(Kernel, IsotropicMixin):
    r"""Matern kernel.

    Covariance function defined by

    .. math::
        :nowrap:

        \begin{equation}
            k(x_0, x_1)
            =
            \frac{1}{\Gamma(\nu) 2^{\nu - 1}}
            \left( \frac{\sqrt{2 \nu}}{l} \lVert x_0 - x_1 \rVert_2 \right)^\nu
            K_\nu \left( \frac{\sqrt{2 \nu}}{l} \lVert x_0 - x_1 \rVert_2 \right),
        \end{equation}

    where :math:`K_\nu` is a modified Bessel function. The Matern kernel generalizes the
    :class:`~probnum.kernels.ExpQuad` kernel via its additional parameter :math:`\nu`
    controlling the smoothness of the function. For :math:`\nu \rightarrow \infty`
    the Matern kernel converges to the :class:`~probnum.kernels.ExpQuad` kernel. A
    Gaussian process with Matern covariance function is :math:`\lceil \nu \rceil - 1`
    times differentiable.

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
    >>> K.matrix(xs)
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
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        if not self.lengthscale > 0:
            raise ValueError(f"Lengthscale l={self.lengthscale} must be positive.")
        self.nu = _utils.as_numpy_scalar(nu)
        if not self.nu > 0:
            raise ValueError(f"Hyperparameter nu={self.nu} must be positive.")

        super().__init__(input_dim=input_dim, output_dim=None)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        dists = self._euclidean_distances(x0, x1)

        # Kernel matrix computation dependent on differentiability
        if self.nu == 0.5:
            return np.exp(-1.0 / self.lengthscale * dists)

        if self.nu == 1.5:
            scaled_dists = -np.sqrt(3) / self.lengthscale * dists
            return (1.0 + scaled_dists) * np.exp(-scaled_dists)

        if self.nu == 2.5:
            scaled_dists = np.sqrt(5) / self.lengthscale * dists
            return (1.0 + scaled_dists + scaled_dists ** 2 / 3.0) * np.exp(
                -scaled_dists
            )

        if self.nu == np.inf:
            return np.exp(-1.0 / (2.0 * self.lengthscale ** 2) * dists ** 2)

        # The modified Bessel function K_nu is not defined for z=0
        dists = np.maximum(dists, np.finfo(dists.dtype).eps)

        scaled_dists = np.sqrt(2 * self.nu) / self.lengthscale * dists
        return (
            2 ** (1.0 - self.nu)
            / scipy.special.gamma(self.nu)
            * scaled_dists ** self.nu
            * scipy.special.kv(self.nu, scaled_dists)
        )
