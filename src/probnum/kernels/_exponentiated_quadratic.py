"""Exponentiated quadratic kernel."""

import functools
from typing import Optional

from probnum import _backend
from probnum import utils as _utils
from probnum.typing import ArrayType, IntArgType, ScalarArgType

from ._kernel import IsotropicMixin, Kernel


class ExpQuad(Kernel, IsotropicMixin):
    r"""Exponentiated quadratic / RBF kernel.

    Covariance function defined by

    .. math ::
        k(x_0, x_1) = \exp \left( -\frac{\lVert x_0 - x_1 \rVert_2^2}{2 l^2} \right).

    This kernel is also known as the squared
    exponential or radial basis function kernel.

    Parameters
    ----------
    input_dim :
        Input dimension of the kernel.
    lengthscale
        Lengthscale :math:`l` of the kernel. Describes the input scale on which the
        process varies.

    See Also
    --------
    RatQuad : Rational quadratic kernel.
    Matern : Matern kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.kernels import ExpQuad
    >>> K = ExpQuad(input_dim=1, lengthscale=0.1)
    >>> xs = np.linspace(0, 1, 3)[:, None]
    >>> K.matrix(xs)
    array([[1.00000000e+00, 3.72665317e-06, 1.92874985e-22],
           [3.72665317e-06, 1.00000000e+00, 3.72665317e-06],
           [1.92874985e-22, 3.72665317e-06, 1.00000000e+00]])
    """

    def __init__(self, input_dim: IntArgType, lengthscale: ScalarArgType = 1.0):
        self.lengthscale = _utils.as_scalar(lengthscale)
        super().__init__(input_dim=input_dim)

    @functools.partial(_backend.jit, static_argnums=(0,))
    def _evaluate(self, x0: ArrayType, x1: Optional[ArrayType]) -> ArrayType:
        if x1 is None:
            return _backend.ones_like(x0[..., 0])

        return _backend.exp(
            -self._squared_euclidean_distances(x0, x1) / (2.0 * self.lengthscale ** 2)
        )
