"""Polynomial kernel."""

from typing import Optional

from probnum import backend
from probnum.backend.typing import IntLike, ScalarLike, ShapeLike

from ._kernel import Kernel


class Polynomial(Kernel):
    r"""Polynomial kernel.

    Covariance function defined by

    .. math ::
        k(x_0, x_1) = (x_0^\top x_1 + c)^q.

    Parameters
    ----------
    input_shape
        Shape of the kernel's input.
    constant
        Constant offset :math:`c`.
    exponent
        Exponent :math:`q` of the polynomial.

    See Also
    --------
    Linear : Linear covariance function.

    Examples
    --------
    >>> from probnum import backend
    >>> from probnum.randprocs.kernels import Polynomial
    >>> K = Polynomial(input_shape=2, constant=1.0, exponent=3)
    >>> xs = backend.asarray([[1, -1], [-1, 0]])
    >>> K.matrix(xs)
    array([[27.,  0.],
           [ 0.,  8.]])
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        constant: ScalarLike = 0.0,
        exponent: IntLike = 1.0,
    ):
        self.constant = backend.asscalar(constant)
        self.exponent = backend.asscalar(exponent)
        super().__init__(input_shape=input_shape)

    @backend.jit_method
    def _evaluate(
        self, x0: backend.Array, x1: Optional[backend.Array] = None
    ) -> backend.Array:
        return (self._euclidean_inner_products(x0, x1) + self.constant) ** self.exponent
