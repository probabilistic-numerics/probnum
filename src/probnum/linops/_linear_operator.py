"""Finite-dimensional linear operators."""

from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse.linalg
import scipy.sparse.linalg.interface

import probnum.utils
from probnum.type import DTypeArgType, ScalarArgType, ShapeArgType

OperandType = Union[np.ndarray, "RandomVariable"]

BinaryOperandType = Union[
    "LinearOperator", ScalarArgType, np.ndarray, scipy.sparse.spmatrix
]


class LinearOperator:
    r"""Composite base class for finite-dimensional linear operators.

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
    shape :
        Matrix dimensions `(M, N)`.
    dtype :
        Data type of the operator.
    matmul :
        Callable which computes the matrix-matrix product :math:`y = A V`, where
        :math:`A` is the linear operator and :math:`V` is an :math:`N \times K` matrix.
        The callable must support broadcasted matrix products, i.e. the argument
        :math:`V` might also be a stack of matrices in which case the broadcasting rules
        of :func:`np.matmul` must apply.
        Note that the argument to this callable is guaranteed to have at least two
        dimensions.
    rmatmul :
        Callable which implements the matrix-matrix product, i.e. :math:`A @ V`, where
        :math:`A` is the linear operator and :math:`V` is a matrix of shape `(N, K)`.
    todense :
        Callable which returns a dense matrix representation of the linear operator as a
        :class:`np.ndarray`. The output of this function must be equivalent to the
        output of :code:`A.matmat(np.eye(N, dtype=A.dtype))`.
    rmatvec :
        Callable which implements the matrix-vector product with the adjoint of the
        operator, i.e. :math:`A^H v`, where :math:`A^H` is the conjugate transpose of
        the linear operator :math:`A` and :math:`v` is a vector of shape `(N,)`.
        This argument will be ignored if `adjoint` is given.
    rmatmat :
        Returns :math:`A^H V`, where :math:`V` is a dense matrix with dimensions (M, K).

    See Also
    --------
    aslinop : Transform into a LinearOperator.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.linops import LinearOperator

    >>> @LinearOperator.broadcast_matvec
    ... def mv(v):
    ...     return np.array([2 * v[0] - v[1], 3 * v[1]])

    >>> A = LinearOperator(shape=(2, 2), dtype=np.float_, matmul=mv)
    >>> A
    <LinearOperator with shape=(2, 2) and dtype=float64>

    >>> A @ np.array([1., 2.])
    array([0., 6.])
    >>> A @ np.ones(2)
    array([1., 3.])
    """

    ndim = 2

    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType,
        *,
        matmul: Callable[[OperandType], OperandType],
        rmatmul: Optional[Callable[[OperandType], OperandType]] = None,
        todense: Optional[Callable[[], np.ndarray]] = None,
        transpose: Optional[Callable[[OperandType], OperandType]] = None,
        adjoint: Optional[Callable[[], "LinearOperator"]] = None,
        hmatmul: Optional[Callable[[OperandType], OperandType]] = None,
        inverse: Optional[Callable[[], "LinearOperator"]] = None,
        rank: Optional[Callable[[], np.intp]] = None,
        eigvals: Optional[Callable[[], np.ndarray]] = None,
        cond: Optional[
            Callable[[Optional[Union[None, int, str, np.floating]]], np.number]
        ] = None,
        det: Optional[Callable[[], np.number]] = None,
        logabsdet: Optional[Callable[[], np.flexible]] = None,
        trace: Optional[Callable[[], np.number]] = None,
    ):
        self.shape = probnum.utils.as_shape(shape, ndim=2)

        # DType
        self.dtype = np.dtype(dtype)

        if not np.issubdtype(self.dtype, np.number):
            raise TypeError("The dtype of a linear operator must be numeric.")

        # Matrix multiplication (self @ x)
        self.__matmul = matmul

        # Reverse matrix multiplication (x @ self)
        if rmatmul is not None:
            self.__rmatmul = rmatmul
        else:
            self.__rmatmul = lambda x: np.swapaxes(
                self.T @ np.swapaxes(x, -2, -1), -2, -1
            )

        # Dense matrix representation
        if todense is not None:
            self.__todense = todense
        else:
            self.__todense = lambda: self @ np.eye(
                self.shape[1], dtype=self.dtype, order="F"
            )

        self._dense = None

        # Transpose and Adjoint
        if transpose is not None:
            self.__transpose = transpose
        elif adjoint is not None or hmatmul is not None:
            # Fast adjoint operator is available
            if np.issubdtype(self.dtype, np.complexfloating):
                self.__transpose = lambda: _TransposedLinearOperator(
                    self, matmul=lambda x: np.conj(self.H @ np.conj(x))
                )
            else:
                # Transpose == adjoint
                self.__transpose = lambda: self.H
        elif rmatmul is not None:
            self.__transpose = lambda: _TransposedLinearOperator(
                self,
                # This is potentially slower than conjugating a vector twice
                matmul=lambda x: np.swapaxes(rmatmul(np.swapaxes(x, -2, -1)), -2, -1),
            )
        else:
            self.__transpose = lambda: _TransposedLinearOperator(self)

        if adjoint is not None:
            self.__adjoint = adjoint
        elif hmatmul is not None:
            self.__adjoint = lambda: _AdjointLinearOperator(self, matmul=hmatmul)
        elif transpose is not None or rmatmul is not None:
            # Fast transpose operator is available
            if np.issubdtype(self.dtype, np.complexfloating):
                self.__adjoint = lambda: _AdjointLinearOperator(
                    self, matmul=lambda x: np.conj(self.T @ np.conj(x))
                )
            else:
                # Adjoint == transpose
                self.__adjoint = lambda: self.T
        else:
            self.__adjoint = lambda: _AdjointLinearOperator(self)

        # Inverse
        if inverse is not None:
            self.__inverse = inverse
        else:
            self.__inverse = lambda: _InverseLinearOperator(self)

        # Matrix properties
        if rank is not None:
            self.__rank = rank
        else:
            self.__rank = lambda: np.linalg.matrix_rank(self.todense())

        if eigvals is not None:
            self.__eigvals = eigvals
        else:
            self.__eigvals = lambda: np.linalg.eigvals(self.todense())

        if cond is not None:
            self.__cond = cond
        else:
            self.__cond = lambda p: np.linalg.cond(self.todense(), p=p)

        if det is not None:
            self.__det = det
        else:
            self.__det = lambda: np.linalg.det(self.todense())

        if logabsdet is not None:
            self.__logabsdet = logabsdet
        else:
            self.__logabsdet = lambda: np.log(np.abs(self.det()))

        if trace is not None:
            self.__trace = trace
        else:
            self.__trace = self._trace_fallback

    @property
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __call__(self, x: OperandType) -> OperandType:
        return self @ x

    def todense(self) -> np.ndarray:
        """Dense matrix representation of the linear operator.

        This method can be computationally very costly depending on the shape of the
        linear operator. Use with caution.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the linear operator.
        """
        if self._dense is None:
            self._dense = self.__todense()

        return self._dense

    ####################################################################################
    # Derived Quantities
    ####################################################################################

    def rank(self) -> np.intp:
        """Rank of the linear operator."""
        return self.__rank()

    def eigvals(self) -> np.ndarray:
        """Eigenvalue spectrum of the linear operator."""
        if not self.is_square:
            raise np.linalg.LinAlgError(
                "Eigenvalues are only defined on square operators"
            )

        return self.__eigvals()

    def cond(self, p=None) -> np.number:
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
        return self.__cond(p)

    def det(self) -> np.number:
        """Determinant of the linear operator."""
        if not self.is_square:
            raise np.linalg.LinAlgError(
                "The determinant is only defined on square operators"
            )

        return self.__det()

    def logabsdet(self) -> np.inexact:
        """Log absolute determinant of the linear operator."""
        if not self.is_square:
            raise np.linalg.LinAlgError(
                "The determinant is only defined on square operators"
            )

        return self.__logabsdet()

    def trace(self) -> np.number:
        """Trace of the linear operator.

        Computes the trace of a square linear operator :math:`\\text{tr}(A) =
        \\sum_{i-1}^n A_ii`.

        Returns
        -------
        trace : float
            Trace of the linear operator.

        Raises
        ------
        LinAlgError : If :meth:`trace` is called on a non-square matrix.
        """
        if not self.is_square:
            raise np.linalg.LinAlgError(
                "The trace is only defined on square operators."
            )

        return self.__trace()

    ####################################################################################
    # Unary Arithmetic
    ####################################################################################

    def __neg__(self) -> "LinearOperator":
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            NegatedLinearOperator,
        )

        return NegatedLinearOperator(self)

    def adjoint(self) -> "LinearOperator":
        return self.__adjoint()

    @property
    def H(self):
        return self.adjoint()

    def transpose(self) -> "LinearOperator":
        """Transpose this linear operator.

        Can be abbreviated self.T instead of self.transpose().
        """
        return self.__transpose()

    @property
    def T(self):
        return self.transpose()

    def inv(self) -> "LinearOperator":
        """Inverse of the linear operator."""
        return self.__inverse()

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
        """Matrix-vector multiplication.

        Performs the operation `y = self @ x` where `self` is an MxN linear operator
        and `x` is a 1-d array or random variable.

        Parameters
        ----------
        x :
            An array or `RandomVariable` with shape `(N,)` or `(N, 1)`.
        Returns
        -------
        y :
            A `np.matrix` or `np.ndarray` or `RandomVariable` with shape `(M,)` or `(M, 1)`,
            depending on the type and shape of the x argument.
        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.
        """

        if isinstance(other, LinearOperator):
            from ._arithmetic import matmul  # pylint: disable=import-outside-toplevel

            return matmul(self, other)
        else:
            x = other

            M, N = self.shape

            if x.ndim == 1 and x.shape == (N,):
                y = self.__matmul(x[:, np.newaxis])
            elif x.ndim > 1 and x.shape[-2] == N:
                y = self.__matmul(x)
            else:
                raise ValueError(
                    f"Dimension mismatch. Expected operand of shape ({N},) or "
                    f"(..., {N}, K), but got {x.shape}."
                )

            assert y.ndim > 1
            assert y.shape[-2] == M
            assert y.shape[-1] == x.shape[-1] if x.ndim > 1 else y.shape[-1] == 1
            assert y.shape[:-2] == x.shape[:-2]

            if x.ndim == 1:
                y = y.reshape(-1)

            return y

    def __rmatmul__(
        self, other: BinaryOperandType
    ) -> Union["LinearOperator", np.ndarray]:
        if isinstance(other, LinearOperator):
            from ._arithmetic import matmul  # pylint: disable=import-outside-toplevel

            return matmul(other, self)
        else:
            x = other

            M, N = self.shape

            if x.ndim == 1 and x.shape == (M,):
                y = self.__rmatmul(x[np.newaxis, :])
            elif x.ndim > 1 and x.shape[-1] == M:
                y = self.__rmatmul(x)
            else:
                raise ValueError(
                    f"Dimension mismatch. Expected operand of shape ({M},) or "
                    f"(..., {M}), but got {x.shape}."
                )

            if x.ndim == 1:
                y = y.reshape(-1)

            assert y.ndim >= 1
            assert y.shape[-1] == N
            assert y.shape[:-1] == x.shape[:-1]

            return y

    ####################################################################################
    # Automatic (r)mat{vec,mul} Broadcasting
    ####################################################################################

    @classmethod
    def broadcast_matvec(
        cls, matvec: Callable[[OperandType], OperandType]
    ) -> Callable[[OperandType], OperandType]:
        def _matmul(x: OperandType) -> OperandType:
            if x.ndim == 2 and x.shape[1] == 1:
                return matvec(x[:, 0])[:, np.newaxis]

            return np.apply_along_axis(matvec, -2, x)

        return _matmul

    @classmethod
    def broadcast_rmatvec(
        cls, rmatvec: Callable[[OperandType], OperandType]
    ) -> Callable[[OperandType], OperandType]:
        def _rmatmul(x: OperandType) -> OperandType:
            if x.ndim == 2 and x.shape[0] == 1:
                return rmatvec(x[0, :])[np.newaxis, :]

            return np.apply_along_axis(rmatvec, -1, x)

        return _rmatmul

    def _trace_fallback(self) -> np.number:
        vec = np.zeros(self.shape[1], dtype=self.dtype)

        vec[0] = 1
        trace = (self @ vec)[0]
        vec[0] = 0

        for i in range(1, self.shape[0]):
            vec[i] = 1
            trace += (self @ vec)[i]
            vec[i] = 0

        return trace

    @property
    def _inexact_dtype(self) -> np.dtype:
        if np.issubdtype(self.dtype, np.inexact):
            return self.dtype
        else:
            return np.double


class _TransposedLinearOperator(LinearOperator):
    """Transposition of a linear operator."""

    def __init__(
        self,
        linop: LinearOperator,
        matmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._linop = linop

        if matmul is None:
            matmul = lambda x: self.todense() @ x

        super().__init__(
            shape=(self._linop.shape[1], self._linop.shape[0]),
            dtype=self._linop.dtype,
            matmul=matmul,
            todense=lambda: self._linop.todense().T.copy(order="C"),
            transpose=lambda: self._linop,
        )

    def inv(self):
        return self._linop.inv().T


class _AdjointLinearOperator(LinearOperator):
    def __init__(
        self,
        linop: LinearOperator,
        matmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._linop = linop

        if matmul is None:
            matmul = lambda x: self.todense() @ x

        super().__init__(
            shape=(self._linop.shape[1], self._linop.shape[0]),
            dtype=self._linop.dtype,
            matmul=matmul,
            todense=lambda: np.conj(self._linop.todense().T).copy(order="C"),
            adjoint=lambda: self._linop,
        )


class _InverseLinearOperator(LinearOperator):
    def __init__(self, linop: LinearOperator):
        if not linop.is_square:
            raise np.linalg.LinAlgError("Only square operators can be inverted.")

        self._linop = linop

        if np.issubdtype(self._linop.dtype, np.inexact):
            dtype = self._linop.dtype
        else:
            dtype = np.double

        super().__init__(
            shape=self._linop.shape,
            dtype=dtype,
            matmul=lambda x: self.todense() @ x,
            todense=lambda: np.linalg.inv(self._linop.todense()),
            inverse=lambda: self._linop,
        )


class MatrixMult(LinearOperator):
    """A linear operator defined via a matrix.

    Parameters
    ----------
    A : array-like or scipy.sparse.spmatrix
        The explicit matrix.
    """

    def __init__(
        self,
        A: Union[np.ndarray, scipy.sparse.spmatrix],
    ):
        shape = A.shape
        dtype = A.dtype

        self.A = A

        if isinstance(self.A, scipy.sparse.spmatrix):
            todense = self.A.toarray
            inverse = lambda: MatrixMult(scipy.sparse.linalg.inv(self.A))
        elif isinstance(self.A, np.ndarray):
            todense = lambda: self.A
            inverse = lambda: MatrixMult(np.linalg.inv(self.A))
        else:
            raise TypeError(
                f"`A` must be a `np.ndarray` or `scipy.sparse.spmatrix`, but a "
                f"`{type(self.A)}` was given."
            )

        transpose = lambda: MatrixMult(self.A.T)

        if np.issubdtype(dtype, np.complexfloating):
            adjoint = lambda: MatrixMult(np.conj(self.A.T))
        else:
            adjoint = transpose

        super().__init__(
            shape,
            dtype,
            matmul=lambda x: A @ x,
            rmatmul=lambda x: x @ A,
            todense=todense,
            transpose=transpose,
            adjoint=adjoint,
            inverse=inverse,
        )

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
        if self.is_square:
            return self.A.diagonal().sum()
        else:
            return super().trace()
