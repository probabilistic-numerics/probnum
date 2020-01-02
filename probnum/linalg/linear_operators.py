"""(Finite dimensional) linear operators."""

import numpy as np
import scipy.sparse.linalg


class LinearOperator(scipy.sparse.linalg.LinearOperator):
    """
    Finite dimensional linear operators.


    """

    def __init__(self, Op=None, explicit=False):
        # todo: document function
        self.explicit = explicit
        if Op is not None:
            self.Op = Op
            self.shape = self.Op.shape
            self.dtype = self.Op.dtype

    def _matvec(self, x):
        if callable(self.Op._matvec):
            return self.Op._matvec(x)

    def _rmatvec(self, x):
        if callable(self.Op._rmatvec):
            return self.Op._rmatvec(x)

    def _matmat(self, X):
        """Matrix-matrix multiplication handler.
        Modified version of scipy _matmat to avoid having trailing dimension
        in col when provided to matvec.
        """
        # TODO: do we need this?
        return np.vstack([self.matvec(col.reshape(-1)) for col in X.T]).T

    def __mul__(self, x):
        y = super().__mul__(x)
        if isinstance(y, scipy.sparse.linalg.LinearOperator):
            y = LinearOperator(y)
        return y

    def __rmul__(self, x):
        return LinearOperator(super().__rmul__(x))

    def __pow__(self, p):
        return LinearOperator(super().__pow__(p))

    def __add__(self, x):
        return LinearOperator(super().__add__(x))

    def __neg__(self):
        return LinearOperator(super().__neg__())

    def __sub__(self, x):
        return LinearOperator(super().__sub__(x))

    def _adjoint(self):
        return LinearOperator(super()._adjoint())

    # TODO: remaining operations (transpose, todense, eigs, cond, ...)
