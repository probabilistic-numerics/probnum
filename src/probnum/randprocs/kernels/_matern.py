"""Matern kernel."""

from typing import Optional

import numpy as np
import scipy.spatial.distance
import scipy.special

from probnum.typing import ScalarLike, ShapeLike
import probnum.utils as _utils

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
    :class:`~probnum.randprocs.kernels.ExpQuad` kernel via its additional parameter
    :math:`\nu` controlling the smoothness of the function. For :math:`\nu \rightarrow
    \infty` the Matern kernel converges to the :class:`~probnum.randprocs.kernels.\
    ExpQuad` kernel. A Gaussian process with Matern covariance function is :math:`\lceil
    \nu \rceil - 1` times differentiable.

    Parameters
    ----------
    input_shape :
        Shape of the kernel's input.
    lengthscale :
        Lengthscale :math:`l` of the kernel. Describes the input scale on which the
        process varies.
    nu :
        Hyperparameter :math:`\nu` controlling differentiability.

    See Also
    --------
    ExpQuad : Exponentiated Quadratic / RBF kernel.
    ProductMatern : Product Matern kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.kernels import Matern
    >>> K = Matern(input_shape=(), lengthscale=0.1, nu=2.5)
    >>> xs = np.linspace(0, 1, 3)
    >>> K.matrix(xs)
    array([[1.00000000e+00, 7.50933789e-04, 3.69569622e-08],
           [7.50933789e-04, 1.00000000e+00, 7.50933789e-04],
           [3.69569622e-08, 7.50933789e-04, 1.00000000e+00]])
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscale: ScalarLike = 1.0,
        nu: ScalarLike = 1.5,
    ):
        self.lengthscale = _utils.as_numpy_scalar(lengthscale)
        if not self.lengthscale > 0:
            raise ValueError(f"Lengthscale l={self.lengthscale} must be positive.")
        self.nu = _utils.as_numpy_scalar(nu)
        if not self.nu > 0:
            raise ValueError(f"Hyperparameter nu={self.nu} must be positive.")

        super().__init__(input_shape=input_shape)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        distances = self._euclidean_distances(x0, x1)

        # Kernel matrix computation dependent on differentiability
        if self.nu == 0.5:
            return np.exp(-1.0 / self.lengthscale * distances)

        if self.nu == 1.5:
            scaled_distances = np.sqrt(3) / self.lengthscale * distances
            return (1.0 + scaled_distances) * np.exp(-scaled_distances)

        if self.nu == 2.5:
            scaled_distances = np.sqrt(5) / self.lengthscale * distances
            return (1.0 + scaled_distances + scaled_distances**2 / 3.0) * np.exp(
                -scaled_distances
            )
        if self.nu == 3.5:
            scaled_distances = np.sqrt(7) / self.lengthscale * distances
            # Using Horner's method speeds up computations substantially
            return (
                1.0
                + (1.0 + (2.0 / 5.0 + scaled_distances / 15.0) * scaled_distances)
                * scaled_distances
            ) * np.exp(-scaled_distances)

        if self.nu == np.inf:
            return np.exp(-1.0 / (2.0 * self.lengthscale**2) * distances**2)

        # The modified Bessel function K_nu is not defined for z=0
        distances = np.maximum(distances, np.finfo(distances.dtype).eps)

        scaled_distances = np.sqrt(2 * self.nu) / self.lengthscale * distances
        return (
            2 ** (1.0 - self.nu)
            / scipy.special.gamma(self.nu)
            * scaled_distances**self.nu
            * scipy.special.kv(self.nu, scaled_distances)
        )
