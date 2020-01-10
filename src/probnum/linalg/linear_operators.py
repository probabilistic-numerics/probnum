"""Finite dimensional linear operators.

This module defines classes and methods that implement finite dimensional linear operators. It can be used to do linear
algebra with (structured) matrices without explicitly representing them in memory. This often allows for the definition
of a more efficient matrix-vector product. Linear operators can be applied, added, multiplied, transposed, and more as
one would expect from matrix algebra.

Several algorithms in the :mod:`probnum.linalg` library are able to operate on :class:`LinearOperator` instances.
"""

import numpy as np
import scipy.sparse.linalg
import warnings
from probnum.probability import RandomVariable
from pylops import LinearOperator, MatrixMult, Identity, Zero, Diagonal, Transpose, Sum, Symmetrize, BlockDiag, \
    Kronecker, Smoothing1D, Smoothing2D, FirstDerivative, SecondDerivative, Laplacian, Gradient, \
    FirstDirectionalDerivative, SecondDirectionalDerivative

__all__ = ["MatrixMult", "Identity", "Zero", "Diagonal", "Transpose", "Sum", "Symmetrize", "BlockDiag", "Kronecker",
           "SymmetricKronecker", "Smoothing1D", "Smoothing2D", "FirstDerivative", "SecondDerivative", "Laplacian",
           "Gradient", "FirstDirectionalDerivative", "SecondDirectionalDerivative", "aslinop"]


# # TODO: multiple inheritance with RandomVariable
# class LinearOperator(scipy.sparse.linalg.LinearOperator):
#     """
#     Finite dimensional linear operators.
#
#     This class provides a way to define finite dimensional linear operators without explicitly constructing a matrix
#     representation. Instead it suffices to define a matrix-vector product and a shape attribute. This avoids unnecessary
#     memory usage and can often be more convenient to derive.
#
#     LinearOperator instances can be multiplied, added and exponentiated. This happens lazily: the result of these
#     operations is a new, composite LinearOperator, that defers linear operations to the original operators and combines
#     the results.
#
#     To construct a concrete LinearOperator, either pass appropriate callables to the constructor of this class, or
#     subclass it.
#
#     A subclass must implement either one of the methods ``_matvec`` and ``_matmat``, and the
#     attributes/properties ``shape`` (pair of integers) and ``dtype`` (may be ``None``). It may call the ``__init__`` on
#     this class to have these attributes validated. Implementing ``_matvec`` automatically implements ``_matmat`` (using
#     a naive algorithm) and vice-versa.
#
#     Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint`` to implement the Hermitian adjoint (conjugate
#     transpose). As with ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or ``_adjoint`` implements the
#     other automatically. Implementing ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for backwards
#     compatibility.
#
#     This class wraps :class:`scipy.sparse.linalg.LinearOperator` to provide support for
#     :class:`~probnum.RandomVariable`.
#
#     Parameters
#     ----------
#     shape : tuple
#         Matrix dimensions (M, N).
#     matvec : callable f(v)
#         Returns :math:`A v`.
#     rmatvec : callable f(v)
#         Returns :math:`A^H v`, where :math:`A^H` is the conjugate transpose of :math:`A`.
#     matmat : callable f(V)
#         Returns :math:`AV`, where :math:`V` is a dense matrix with dimensions (N, K).
#     dtype : dtype
#         Data type of the operator.
#     rmatmat : callable f(V)
#         Returns :math:`A^H V`, where :math:`V` is a dense matrix with dimensions (M, K).
#
#     See Also
#     --------
#     aslinearoperator : Transform into a LinearOperator.
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> from probnum.linalg import LinearOperator
#     >>> def mv(v):
#     ...     return np.array([2*v[0], 3*v[1]])
#     ...
#     >>> A = LinearOperator((2,2), matvec=mv)
#     >>> A
#     <2x2 _CustomLinearOperator with dtype=float64>
#     >>> A.matvec(np.ones(2))
#     array([ 2.,  3.])
#     >>> A * np.ones(2)
#     array([ 2.,  3.])
#
#     """
#
#     def __new__(cls, *args, **kwargs):
#         if cls is LinearOperator:
#             # Operate as _CustomLinearOperator factory.
#             return super(LinearOperator, cls).__new__(_CustomLinearOperator)
#         else:
#             obj = super(LinearOperator, cls).__new__(cls)
#
#             if type(obj)._matvec == LinearOperator._matvec and type(obj)._matmat == LinearOperator._matmat:
#                 warnings.warn("LinearOperator subclass should implement"
#                               " at least one of _matvec and _matmat.",
#                               category=RuntimeWarning, stacklevel=2)
#
#             return obj
#
#     def __init__(self, dtype, shape):
#         super().__init__(dtype=dtype, shape=shape)
#         # todo: how should this constructor work and is it necessary to have our own?
#         # todo: should we only allow subclassing of LinearOperator?
#
#     # self.explicit = explicit
#     # if Op is not None:
#     #     self.Op = Op
#     #     self.shape = self.Op.shape
#     #     self.dtype = self.Op.dtype
#     #
#     # def _matvec(self, x):
#     #     if callable(self.Op._matvec):
#     #         return self.Op._matvec(x)
#     #
#     # def _rmatvec(self, x):
#     #     if callable(self.Op._rmatvec):
#     #         return self.Op._rmatvec(x)
#     #
#     # def _matmat(self, X):
#     #     """Matrix-matrix multiplication handler.
#     #     Modified version of scipy _matmat to avoid having trailing dimension
#     #     in col when provided to matvec.
#     #     """
#     #     # TODO: do we need this?
#     #     return np.vstack([self.matvec(col.reshape(-1)) for col in X.T]).T
#     #
#     # def __mul__(self, x):
#     #     y = super().__mul__(x)
#     #     if isinstance(y, scipy.sparse.linalg.LinearOperator):
#     #         y = LinearOperator(y)
#     #     return y
#     #
#     # def __rmul__(self, x):
#     #     return LinearOperator(super().__rmul__(x))
#     #
#     # def __pow__(self, p):
#     #     return LinearOperator(super().__pow__(p))
#     #
#     # def __add__(self, x):
#     #     return LinearOperator(super().__add__(x))
#     #
#     # def __neg__(self):
#     #     return LinearOperator(super().__neg__())
#     #
#     # def __sub__(self, x):
#     #     return LinearOperator(super().__sub__(x))
#     #
#     # def _adjoint(self):
#     #     return LinearOperator(super()._adjoint())
#     #
#     # def _transpose(self):
#     #     return LinearOperator(super()._transpose())
#     # TODO: override arithmetic operations to make sure that arithmetic operations with random variables work as intended
#
#     def todense(self):
#         """
#         Dense matrix representation of the linear operator.
#
#         This operation can be computationally costly depending on the size of the operator.
#
#         Returns
#         -------
#         matrix : ndarray
#             Dense matrix representation.
#
#         """
#         # needed if self is a _SumLinearOperator or _ProductLinearOperator
#         linop = LinearOperator(self)
#         identity = np.eye(self.shape[1], dtype=self.dtype)
#         return linop.matmat(identity)
#
#     # TODO: implement operations (eigs, cond, det, logabsdet, trace, ...)
#
#
# # todo avoid code duplication and use scipy implementation instead
# class _CustomLinearOperator(LinearOperator):
#     """Linear operator defined in terms of user-specified operations."""
#
#     def __init__(self, shape, matvec, rmatvec=None, matmat=None,
#                  dtype=None, rmatmat=None):
#         super(_CustomLinearOperator, self).__init__(dtype, shape)
#
#         self.args = ()
#
#         self.__matvec_impl = matvec
#         self.__rmatvec_impl = rmatvec
#         self.__rmatmat_impl = rmatmat
#         self.__matmat_impl = matmat
#
#         self._init_dtype()
#
#     def _matmat(self, X):
#         if self.__matmat_impl is not None:
#             return self.__matmat_impl(X)
#         else:
#             return super(_CustomLinearOperator, self)._matmat(X)
#
#     def _matvec(self, x):
#         return self.__matvec_impl(x)
#
#     def _rmatvec(self, x):
#         func = self.__rmatvec_impl
#         if func is None:
#             raise NotImplementedError("rmatvec is not defined")
#         return self.__rmatvec_impl(x)
#
#     def _rmatmat(self, X):
#         if self.__rmatmat_impl is not None:
#             return self.__rmatmat_impl(X)
#         else:
#             return super(_CustomLinearOperator, self)._rmatmat(X)
#
#     def _adjoint(self):
#         return _CustomLinearOperator(shape=(self.shape[1], self.shape[0]),
#                                      matvec=self.__rmatvec_impl,
#                                      rmatvec=self.__matvec_impl,
#                                      matmat=self.__rmatmat_impl,
#                                      rmatmat=self.__matmat_impl,
#                                      dtype=self.dtype)
#
#
# class Identity(LinearOperator):
#     """
#     The identity operator.
#     """
#
#     def __init__(self, shape, dtype=None):
#         if shape[0] != shape[1]:
#             raise ValueError("The identity operator must be square.")
#         super(Identity, self).__init__(dtype=dtype, shape=shape)
#
#     def _matvec(self, x):
#         return x
#
#     def _rmatvec(self, x):
#         return x
#
#     def _rmatmat(self, x):
#         return x
#
#     def _matmat(self, x):
#         return x
#
#     def _adjoint(self):
#         return self
#
#     def todense(self):
#         return np.eye(N=self.shape[0], M=self.shape[1])
#
#
# # TODO: replace this with own implementation?
# class Matrix(scipy.sparse.linalg.interface.MatrixLinearOperator, LinearOperator):
#     """
#     A linear operator defined via a matrix.
#
#     Parameters
#     ----------
#     A : array-like or scipy.sparse.spmatrix
#         The explicit matrix.
#     """
#
#     def __init__(self, A):
#         super().__init__(A=A)
#

class Kronecker(LinearOperator):
    """
    Kronecker product of two linear operators.

    The Kronecker product [1]_ :math:`A \\otimes B` of two linear operators :math:`A` and :math:`B` is given by

    .. math::
        A \\otimes B = \\begin{bmatrix}
            A_{11} B   &  \\dots   & A_{1 m_1} B  \\\\
            \\vdots     &  \\ddots  & \\vdots \\\\
            A_{n_11} B &  \\dots   & A_{n_1 m_1} B
        \\end{bmatrix}

    where :math:`A_{ij}v=A(v_j e_i)`, where :math:`e_i` is the :math:`i^{\\text{th}}` unit vector. The result is a new linear
    operator mapping from :math:`\\mathbb{R}^{n_1n_2}` to :math:`\\mathbb{R}^{m_1m_2}`. By recognizing that
    :math:`(A \\otimes B)\\operatorname{vec}(X) = AXB^{\\top}`, the Kronecker product can be understood as "translation"
    between matrix multiplication and (row-wise) vectorization.

    .. [1] Van Loan, C. F., The ubiquitous Kronecker product, *Journal of Computational and Applied Mathematics*, 2000,
            123, 85-100

    Parameters
    ----------
    A : LinearOperator
        The first operator.
    B : LinearOperator
        The second operator.
    dtype : dtype
        Data type of the operator.

    See Also
    --------
    SymmetricKronecker : The symmetric Kronecker product of two linear operators.

    """

    # todo: extend this to list of operators
    def __init__(self, A, B, dtype=None):
        self.A = A
        self.B = B
        super().__init__(dtype=dtype, shape=(self.A.shape[0] * self.B.shape[0],
                                             self.A.shape[1] * self.B.shape[1]))

    def _matvec(self, x):
        """
        Efficient multiplication via (A x B)vec(X) = vec(AXB^T) where vec is the row-wise vectorization operator.
        """
        x = x.reshape(self.A.shape[1], self.B.shape[1])
        y = self.B.matmat(x.T).T
        return self.A.matmat(y).ravel()

    def _rmatvec(self, x):
        x = x.reshape(self.A.shape[0], self.B.shape[0])
        y = self.B.H.matmat(x.T).T
        return self.A.H.matmat(y).ravel()


class SymmetricKronecker(LinearOperator):
    """
    Symmetric Kronecker product of two linear operators.

    The symmetric Kronecker product [1]_ :math:`A \\otimes_{s} B` of two square linear operators :math:`A` and
    :math:`B` maps a symmetric linear operator :math:`X` to :math:`\\mathbb{R}^{\\frac{1}{2}n (n+1)}`. It is given by

    .. math::
        (A \\otimes_{s} B)\\operatorname{svec}(X) = \\frac{1}{2} \\operatorname{svec}(AXB^{\\top} + BXA^{\\top})

    where :math:`\\operatorname{svec}(X) = (X_{11}, \\sqrt{2} X_{12}, \\dots, X_{1n}, X_{22}, \\sqrt{2} X_{23},
    \\dots, \\sqrt{2}X_{2n}, \\dots X_{nn})^{\\top}` is the (row-wise, normalized) symmetric stacking operator.

    .. [1] Van Loan, C. F., The ubiquitous Kronecker product, *Journal of Computational and Applied Mathematics*, 2000,
            123, 85-100

    Note
    ----
    The symmetric Kronecker product has a symmetric matrix representation if both :math:`A` and :math:`B` are symmetric.

    See Also
    --------
    Kronecker : The Kronecker product of two linear operators.
    """

    def __init__(self, A, B, dtype=None):
        raise NotImplementedError
    # TODO: Make sure definition matches the one in "Hennig, P. and Osborne M. A., Probabilistic Numerics, 2020"


def aslinop(A):
    """
    Return `A` as a :class:`LinearOperator`.

    Parameters
    ----------
    A : array-like or LinearOperator or RandomVariable or object
        Argument to be represented as a linear operator. When `A` is an object it needs to have the attributes `.shape`
        and `.matvec`.

    Notes
    -----
    If `A` has no `.dtype` attribute, the data type is determined by calling
    :func:`LinearOperator.matvec()` - set the `.dtype` attribute to prevent this
    call upon the linear operator creation.

    See Also
    --------
    LinearOperator : Class representing linear operators.

    Examples
    --------
    >>> from probnum.linalg import aslinop
    >>> M = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
    >>> aslinop(M)
    <2x3 MatrixLinearOperator with dtype=int32>
    """
    if isinstance(A, RandomVariable):
        # TODO: aslinearoperator also for random variables; change docstring example;
        #  not needed if LinearOperator inherits from RandomVariable
        raise NotImplementedError
    elif isinstance(A, (np.ndarray, scipy.sparse.spmatrix)):
        return Matrix(A=A)
    else:
        return LinearOperator(scipy.sparse.linalg.aslinearoperator(A))
