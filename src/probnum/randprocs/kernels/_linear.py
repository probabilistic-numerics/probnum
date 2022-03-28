"""Linear covariance function."""

from __future__ import annotations

from typing import Optional

from probnum import backend
from probnum.typing import ArrayType, ScalarLike, ShapeLike

from ._kernel import Kernel


class Linear(Kernel):
    r"""Linear kernel.

    Linear covariance function defined by

    .. math ::
        k(x_0, x_1) = x_0^\top x_1 + c.

    Parameters
    ----------
    input_shape
        Shape of the kernel's input.
    constant
        Constant offset :math:`c`.

    See Also
    --------
    Polynomial : Polynomial covariance function.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.randprocs.kernels import Linear
    >>> K = Linear(input_shape=2)
    >>> xs = np.array([[1, 2], [2, 3]])
    >>> K.matrix(xs)
    array([[ 5.,  8.],
           [ 8., 13.]])
    """

    def __init__(self, input_shape: ShapeLike, constant: ScalarLike = 0.0):
        self.constant = backend.asscalar(constant)
        super().__init__(input_shape=input_shape)

    @backend.jit_method
    def _evaluate(self, x0: ArrayType, x1: Optional[ArrayType]) -> ArrayType:
        return self._euclidean_inner_products(x0, x1) + self.constant
