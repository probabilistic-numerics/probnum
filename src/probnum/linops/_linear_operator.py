"""Finite-dimensional linear operators."""

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse.linalg
import scipy.sparse.linalg.interface

import probnum.utils
from probnum.type import DTypeArgType, ScalarArgType, ShapeArgType

BinaryOperandType = Union[
    "LinearOperator", ScalarArgType, np.ndarray, scipy.sparse.spmatrix
]


class LinearOperator(scipy.sparse.linalg.LinearOperator):
    """Composite base class for finite-dimensional linear operators.

    This class provides a way to define finite-dimensional linear operators without
    explicitly constructing a matrix representation. Instead it suffices to define a
    matrix-vector product and a shape attribute. This avoids unnecessary memory usage
    and can often be more convenient to derive.

    :class:`LinearOperator` instances can be multiplied, added and exponentiated. This
    happens lazily: the result of these operations is a new, composite
    :class:`LinearOperator`, that defers linear operations to the original operators and
    combines the results.

    To construct a concrete class:`LinearOperator`, either pass appropriate callables to
    the constructor of this class, or subclass it.

    A subclass must implement either one of the methods ``_matvec`` and ``_matmat``, and
    the attributes/properties ``shape`` (pair of integers) and ``dtype`` (may be
    ``None``). It may call the ``__init__`` on this class to have these attributes
    validated. Implementing ``_matvec`` automatically implements ``_matmat`` (using a
    naive algorithm) and vice-versa.

    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint`` to implement the
    Hermitian adjoint (conjugate transpose). As with ``_matvec`` and ``_matmat``,
    implementing either ``_rmatvec`` or ``_adjoint`` implements the other automatically.
    Implementing ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for backwards
    compatibility.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns :math:`A v`.
    rmatvec : callable f(v)
        Returns :math:`A^H v`, where :math:`A^H` is the conjugate transpose of
        :math:`A`.
    matmat : callable f(V)
        Returns :math:`AV`, where :math:`V` is a dense matrix with dimensions (N, K).
    dtype : dtype
        Data type of the operator.
    rmatmat : callable f(V)
        Returns :math:`A^H V`, where :math:`V` is a dense matrix with dimensions (M, K).

    See Also
    --------
    aslinop : Transform into a LinearOperator.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.linops import LinearOperator
    >>> def mv(v):
    ...     return np.array([2 * v[0] - v[1], 3 * v[1]])
    ...
    >>> A = LinearOperator(shape=(2, 2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.array([1., 2.]))
    array([0., 6.])
    >>> A @ np.ones(2)
    array([1., 3.])
    """

    def __new__(cls, *args, **kwargs):
        if cls is LinearOperator:
            # _CustomLinearOperator factory
            return super().__new__(_CustomLinearOperator)
        else:
            obj = super().__new__(cls)

            if (
                type(obj)._matvec == scipy.sparse.linalg.LinearOperator._matvec
                and type(obj)._matmat == scipy.sparse.linalg.LinearOperator._matmat
            ):
                warnings.warn(
                    "LinearOperator subclass should implement"
                    " at least one of _matvec and _matmat.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )

            return obj

    # The below methods are overloaded to allow dot products with random variables
    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        Ax : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x.
        """
        return self @ x

    def matvec(self, x):
        """Matrix-vector multiplication. Performs the operation y=A*x where A is an MxN
        linear operator and x is a 1-d array or random variable.

        Parameters
        ----------
        x : {matrix, ndarray, RandomVariable}
            An array or RandomVariable with shape (N,) or (N,1).
        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray or RandomVariable with shape (M,) or (M,1) depending
            on the type and shape of the x argument.
        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.
        """
        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError("Dimension mismatch.")

        y = self._matvec(x)

        if isinstance(x, np.matrix):
            y = scipy.sparse.sputils.asmatrix(y)

        if isinstance(x, (np.matrix, np.ndarray)):
            if x.ndim == 1:
                y = y.reshape(M)
            elif x.ndim == 2:
                y = y.reshape(M, 1)
            else:
                raise ValueError("Invalid shape returned by user-defined matvec().")
        # TODO: can be shortened once RandomVariable implements a reshape method
        elif y.shape[0] != M:
            raise ValueError("Invalid shape returned by user-defined matvec().")

        return y

    def todense(self):
        """Dense matrix representation of the linear operator.

        This method can be computationally very costly depending on the shape of the
        linear operator. Use with caution.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the linear operator.
        """
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    # TODO: implement operations (eigs, cond, det, logabsdet, trace, ...)
    def rank(self):
        """Rank of the linear operator."""
        raise NotImplementedError

    def eigvals(self):
        """Eigenvalue spectrum of the linear operator."""
        raise NotImplementedError

    def cond(self, p=None):
        """Compute the condition number of the linear operator.

        The condition number of the linear operator with respect to the ``p`` norm. It
        measures how much the solution :math:`x` of the linear system :math:`Ax=b`
        changes with respect to small changes in :math:`b`.

        Parameters
        ----------
        p : {None, 1, , 2, , inf, 'fro'}, optional
            Order of the norm:

            =======  ============================
            p        norm for matrices
            =======  ============================
            None     2-norm, computed directly via singular value decomposition
            'fro'    Frobenius norm
            np.inf   max(sum(abs(x), axis=1))
            1        max(sum(abs(x), axis=0))
            2        2-norm (largest sing. value)
            =======  ============================

        Returns
        -------
        cond : {float, inf}
            The condition number of the linear operator. May be infinite.
        """
        raise NotImplementedError

    def det(self):
        """Determinant of the linear operator."""
        raise NotImplementedError

    def logabsdet(self):
        """Log absolute determinant of the linear operator."""
        raise NotImplementedError

    def trace(self):
        """Trace of the linear operator.

        Computes the trace of a square linear operator :math:`\\text{tr}(A) =
        \\sum_{i-1}^n A_ii`.

        Returns
        -------
        trace : float
            Trace of the linear operator.

        Raises
        ------
        ValueError : If :meth:`trace` is called on a non-square matrix.
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError("The trace is only defined for square linear operators.")
        else:
            _identity = np.eye(self.shape[0])
            trace = 0.0
            for i in range(self.shape[0]):
                trace += np.squeeze(
                    _identity[np.newaxis, i, :]
                    @ self.matvec(_identity[i, :, np.newaxis])
                )
            return trace

    ####################################################################################
    # Unary Arithmetic
    ####################################################################################

    def __neg__(self) -> "LinearOperator":
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            NegatedLinearOperator,
        )

        return NegatedLinearOperator(self)

    def transpose(self) -> "LinearOperator":
        """Transpose this linear operator.

        Can be abbreviated self.T instead of self.transpose().
        """
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            TransposedLinearOperator,
        )

        return TransposedLinearOperator(self)

    @property
    def T(self):
        return self.transpose()

    def inv(self) -> "LinearOperator":
        """Inverse of the linear operator."""
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            InverseLinearOperator,
        )

        return InverseLinearOperator(self)

    ####################################################################################
    # Binary Arithmetic
    ####################################################################################

    __array_ufunc__ = None
    """
    This prevents numpy from calling elementwise arithmetic operations allowing
    expressions like `y = np.array([1, 1]) + linop` to call the arithmetic operations
    defined by `LinearOperator` instead of elementwise. Thus no array of
    `LinearOperator`s but a `LinearOperator` with the correct shape is returned.
    """

    def __add__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import add  # pylint: disable=import-outside-toplevel

        return add(self, other)

    def __radd__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import add  # pylint: disable=import-outside-toplevel

        return add(other, self)

    def __sub__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import sub  # pylint: disable=import-outside-toplevel

        return sub(self, other)

    def __rsub__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import sub  # pylint: disable=import-outside-toplevel

        return sub(other, self)

    def __mul__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import mul  # pylint: disable=import-outside-toplevel

        return mul(self, other)

    def __rmul__(self, other: BinaryOperandType) -> "LinearOperator":
        from ._arithmetic import mul  # pylint: disable=import-outside-toplevel

        return mul(other, self)

    def __matmul__(
        self, other: BinaryOperandType
    ) -> Union["LinearOperator", np.ndarray]:
        if isinstance(other, LinearOperator):
            from ._arithmetic import matmul  # pylint: disable=import-outside-toplevel

            return matmul(self, other)
        else:
            if len(other.shape) == 1 or len(other.shape) == 2 and other.shape[1] == 1:
                return self.matvec(other)
            elif len(other.shape) == 2:
                return self.matmat(other)
            else:
                raise ValueError(
                    f"Expected 1-d or 2-d array, matrix or random variable, got "
                    f"{other}."
                )

    def __rmatmul__(
        self, other: BinaryOperandType
    ) -> Union["LinearOperator", np.ndarray]:
        # TODO: rmatvec and rmatmat
        return NotImplemented


class _CustomLinearOperator(
    scipy.sparse.linalg.interface._CustomLinearOperator, LinearOperator
):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(
        self, shape, matvec, rmatvec=None, matmat=None, rmatmat=None, dtype=None
    ):
        super().__init__(
            shape=shape,
            matvec=matvec,
            rmatvec=rmatvec,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=dtype,
        )


class Diagonal(LinearOperator):
    """A linear operator representing the diagonal from another linear operator.

    Parameters
    ----------
    Op : LinearOperator
        Linear operator of which to represent the diagonal.
    """

    # TODO: should this be an operator itself or a function of a LinearOperator?
    #   - a function allows subclasses (e.g. MatrixMult) to implement more efficient
    # versions than n products e_i A e_i
    def __init__(self, Op):
        # pylint: disable=super-init-not-called
        raise NotImplementedError


class ScalarMult(LinearOperator):
    """A linear operator representing scalar multiplication.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    scalar : float
        Scalar to multiply by.
    """

    def __init__(
        self,
        shape: ShapeArgType,
        scalar: ScalarArgType,
        dtype: Optional[DTypeArgType] = None,
    ):
        self._scalar = probnum.utils.as_numpy_scalar(scalar, dtype=dtype)

        super().__init__(shape=probnum.utils.as_shape(shape), dtype=self._scalar.dtype)

    @property
    def scalar(self):
        return self._scalar

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._scalar * x

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return x.T * self._scalar

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        return self._scalar * X

    def todense(self):
        return np.eye(self.shape[0]) * self._scalar

    def inv(self) -> LinearOperator:
        return ScalarMult(shape=self.shape, scalar=1 / self._scalar)

    # Properties
    def rank(self) -> int:
        return min(self.shape)

    def eigvals(self):
        return np.ones(self.shape[0]) * self._scalar

    def cond(self, p=None):
        return 1

    def det(self):
        return self._scalar ** self.shape[0]

    def logabsdet(self):
        return np.log(np.abs(self._scalar))

    def trace(self):
        return self._scalar * self.shape[0]


class Identity(ScalarMult):
    """The identity operator.

    Parameters
    ----------
    shape : int or tuple
        Shape of the identity operator.
    """

    def __init__(self, shape):
        # Check shape
        if np.isscalar(shape):
            _shape = (shape, shape)
        elif shape[0] != shape[1]:
            raise ValueError("The identity operator must be square.")
        else:
            _shape = shape
        # Initiator of super class
        super().__init__(shape=_shape, scalar=1.0)

    def todense(self):
        return np.eye(self.shape[0])

    def inv(self):
        return self

    # Properties
    def rank(self):
        return self.shape[0]

    def eigvals(self):
        return np.ones(self.shape[0])

    def cond(self, p=None):
        return 1

    def det(self):
        return 1.0

    def logabsdet(self):
        return 0.0

    def trace(self):
        return self.shape[0]


class MatrixMult(scipy.sparse.linalg.interface.MatrixLinearOperator, LinearOperator):
    """A linear operator defined via a matrix.

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix
        The explicit matrix.
    """

    def __init__(self, A):
        super().__init__(A=A)

    def _matvec(self, x):
        return self.A @ x  # Needed to call __matmul__ instead of np.dot or np.matmul

    def _matmat(self, X):
        return self.A @ X

    def todense(self):
        if isinstance(self.A, scipy.sparse.spmatrix):
            return self.A.todense()
        else:
            return np.asarray(self.A)

    def inv(self):
        if isinstance(self.A, scipy.sparse.spmatrix):
            invmat = scipy.sparse.linalg.inv(self.A)
        else:
            invmat = np.linalg.inv(self.A)
        return MatrixMult(invmat)

    # Arithmetic operations
    # TODO: perform arithmetic operations between MatrixMult operators explicitly

    # Properties
    def rank(self):
        return np.linalg.matrix_rank(self.A)

    def eigvals(self):
        return np.linalg.eigvals(self.A)

    def cond(self, p=None):
        return np.linalg.cond(self.A, p=p)

    def det(self):
        return np.linalg.det(self.A)

    def logabsdet(self):
        _sign, logdet = np.linalg.slogdet(self.A)
        return logdet

    def trace(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("The trace is only defined for square linear operators.")
        else:
            return np.trace(self.A)
