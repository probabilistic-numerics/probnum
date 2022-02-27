"""Exponentiated quadratic kernel."""

from typing import Optional

from probnum import backend
from probnum.typing import IntLike, ScalarLike, ShapeLike

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
    input_shape :
        Shape of the kernel's input.
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
    >>> from probnum.randprocs.kernels import ExpQuad
    >>> K = ExpQuad(input_shape=(), lengthscale=0.1)
    >>> xs = np.linspace(0, 1, 3)
    >>> K.matrix(xs)
    array([[1.00000000e+00, 3.72665317e-06, 1.92874985e-22],
           [3.72665317e-06, 1.00000000e+00, 3.72665317e-06],
           [1.92874985e-22, 3.72665317e-06, 1.00000000e+00]])
    """

    def __init__(self, input_shape: ShapeLike, lengthscale: ScalarLike = 1.0):
        self.lengthscale = backend.as_scalar(lengthscale)
        super().__init__(input_shape=input_shape)

    @backend.jit_method
    def _evaluate(
        self, x0: backend.ndarray, x1: Optional[backend.ndarray]
    ) -> backend.ndarray:
        if x1 is None:
            return backend.ones_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return backend.exp(
            -self._squared_euclidean_distances(x0, x1) / (2.0 * self.lengthscale**2)
        )
