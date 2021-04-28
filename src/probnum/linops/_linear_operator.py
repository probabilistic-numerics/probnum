"""Finite-dimensional linear operators."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.sparse.linalg
import scipy.sparse.linalg.interface

import probnum.utils
from probnum.type import DTypeArgType, ScalarArgType, ShapeArgType

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

    def __init__(
        self,
        shape: ShapeArgType,
        dtype: DTypeArgType,
        *,
        matmul: Callable[[np.ndarray], np.ndarray],
        rmatmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        apply: Callable[[np.ndarray, int], np.ndarray] = None,
        todense: Optional[Callable[[], np.ndarray]] = None,
        transpose: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        adjoint: Optional[Callable[[], "LinearOperator"]] = None,
        hmatmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
        self._shape = probnum.utils.as_shape(shape, ndim=2)

        # DType
        self._dtype = np.dtype(dtype)

        if not np.issubdtype(self._dtype, np.number):
            raise TypeError("The dtype of a linear operator must be numeric.")

        # Matrix multiplication (self @ x)
        self.__matmul = matmul

        # Reverse matrix multiplication (x @ self)
        if rmatmul is not None:
            self.__rmatmul = rmatmul
        else:
            self.__rmatmul = lambda x: (self.T)(x, axis=-1)

        # __call__
        if apply is not None:
            self.__apply = apply
        else:
            self.__apply = lambda x, axis: np.swapaxes(
                self @ np.swapaxes(x, axis, -2), -2, axis
            )

        # Dense matrix representation
        if todense is not None:
            self.__todense = todense
        else:
            self.__todense = lambda: self @ np.eye(
                self.shape[1], dtype=self._dtype, order="F"
            )

        # Transpose and Adjoint
        if transpose is not None:
            self.__transpose = transpose
        elif adjoint is not None or hmatmul is not None:
            # Fast adjoint operator is available
            if np.issubdtype(self._dtype, np.complexfloating):
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
                matmul=lambda x: rmatmul(x[..., np.newaxis])[..., :],
            )
        else:
            self.__transpose = lambda: _TransposedLinearOperator(self)

        if adjoint is not None:
            self.__adjoint = adjoint
        elif hmatmul is not None:
            self.__adjoint = lambda: _AdjointLinearOperator(self, matmul=hmatmul)
        elif transpose is not None or rmatmul is not None:
            # Fast transpose operator is available
            if np.issubdtype(self._dtype, np.complexfloating):
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
            self.__rank = lambda: np.linalg.matrix_rank(self.todense(cache=False))

        if eigvals is not None:
            self.__eigvals = eigvals
        else:
            self.__eigvals = lambda: np.linalg.eigvals(self.todense(cache=False))

        if cond is not None:
            self.__cond = cond
        else:
            self.__cond = lambda p: np.linalg.cond(self.todense(cache=False), p=p)

        if det is not None:
            self.__det = det
        else:
            self.__det = lambda: np.linalg.det(self.todense(cache=False))

        if logabsdet is not None:
            self.__logabsdet = logabsdet
        else:
            self.__logabsdet = self._logabsdet_fallback

        if trace is not None:
            self.__trace = trace
        else:
            self.__trace = self._trace_fallback

        # Caches
        self.__todense_cache = None

        self.__rank_cache = None
        self.__eigvals_cache = None
        self.__cond_cache = {}
        self.__det_cache = None
        self.__logabsdet_cache = None
        self.__trace_cache = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __call__(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        if axis is not None and (axis < -x.ndim or axis >= x.ndim):
            raise ValueError(
                f"Axis {axis} is out-of-bounds for operand of shape {np.shape(x)}."
            )

        if x.ndim == 1:
            return self @ x
        elif x.ndim > 1:
            if axis is None:
                axis = -1

            if axis < 0:
                axis += x.ndim

            if axis == (x.ndim - 1):
                return (self @ x[..., np.newaxis])[..., 0]
            elif axis == (x.ndim - 2):
                return self @ x
            else:
                return self.__apply(x, axis)
        else:
            raise ValueError("The operand must be at least one dimensional.")

    def astype(
        self,
        dtype: DTypeArgType,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> "LinearOperator":
        dtype = np.dtype(dtype)

        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"Cannot cast linear operator from {self.dtype} to {dtype} "
                f"according to the rule {casting}"
            )

        if not subok:
            raise NotImplementedError(
                "Setting `subok` to `False` is not supported for linear operators"
            )

        return self._astype(dtype, order, casting, copy)

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "LinearOperator":
        if self.dtype == dtype and not copy:
            return self
        else:
            return _TypeCastLinearOperator(self, dtype, order, casting, copy)

    def todense(self, cache: bool = True) -> np.ndarray:
        """Dense matrix representation of the linear operator.

        This method can be computationally very costly depending on the shape of the
        linear operator. Use with caution.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the linear operator.
        """
        if self.__todense_cache is None:
            if not cache:
                return self.__todense()

            self.__todense_cache = self.__todense()

        return self.__todense_cache

    ####################################################################################
    # Derived Quantities
    ####################################################################################

    def rank(self) -> np.intp:
        """Rank of the linear operator."""
        if self.__rank_cache is None:
            self.__rank_cache = self.__rank()

        return self.__rank_cache

    def eigvals(self) -> np.ndarray:
        """Eigenvalue spectrum of the linear operator."""
        if self.__eigvals_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "Eigenvalues are only defined on square operators"
                )

            self.__eigvals_cache = self.__eigvals()
            self.__eigvals_cache.setflags(write=False)

        return self.__eigvals_cache

    def cond(self, p=None) -> np.inexact:
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
        if p not in self.__cond_cache:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The condition number is only defined on square operators"
                )

            self.__cond_cache[p] = self.__cond(p)

        return self.__cond_cache[p]

    def det(self) -> np.inexact:
        """Determinant of the linear operator."""
        if self.__det_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The determinant is only defined on square operators"
                )

            self.__det_cache = self.__det()

        return self.__det_cache

    def logabsdet(self) -> np.inexact:
        """Log absolute determinant of the linear operator."""
        if self.__logabsdet_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The determinant is only defined on square operators"
                )

            self.__logabsdet_cache = self.__logabsdet()

        return self.__logabsdet_cache

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
        if self.__trace_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The trace is only defined on square operators."
                )

            self.__trace_cache = self.__trace()

        return self.__trace_cache

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
    # Automatic `(r)mat{vec,mat}`` to `(r)matmul` Broadcasting
    ####################################################################################

    @classmethod
    def broadcast_matvec(
        cls, matvec: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def _matmul(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2 and x.shape[1] == 1:
                return matvec(x[:, 0])[:, np.newaxis]

            return np.apply_along_axis(matvec, -2, x)

        return _matmul

    @classmethod
    def broadcast_matmat(
        cls, matmat: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def _matmul(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2:
                return matmat(x)

            return _apply_to_matrix_stack(matmat, x)

        return _matmul

    @classmethod
    def broadcast_rmatvec(
        cls, rmatvec: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def _rmatmul(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2 and x.shape[0] == 1:
                return rmatvec(x[0, :])[np.newaxis, :]

            return np.apply_along_axis(rmatvec, -1, x)

        return _rmatmul

    @classmethod
    def broadcast_rmatmat(
        cls, rmatmat: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def _rmatmul(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2:
                return rmatmat(x)

            return _apply_to_matrix_stack(rmatmat, x)

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

    def _logabsdet_fallback(self) -> np.inexact:
        if self.det() == 0:
            return probnum.utils.as_numpy_scalar(-np.inf, dtype=self._inexact_dtype)
        else:
            return np.log(np.abs(self.det()))

    @property
    def _inexact_dtype(self) -> np.dtype:
        if np.issubdtype(self.dtype, np.inexact):
            return self.dtype
        else:
            return np.double


def _apply_to_matrix_stack(
    mat_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray
) -> np.ndarray:
    idcs = np.ndindex(x.shape[:-2])

    # Shape and dtype inference
    idx0 = next(idcs)
    y0 = mat_fn(x[idx0])

    # Result buffer
    y = np.empty(x.shape[:-2] + y0.shape, dtype=y0.dtype)

    # Fill buffer
    y[idx0] = y0

    for idx in idcs:
        y[idx] = mat_fn(x[idx])

    return y


class _TransposedLinearOperator(LinearOperator):
    """Transposition of a linear operator."""

    def __init__(
        self,
        linop: LinearOperator,
        matmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._linop = linop

        if matmul is None:
            matmul = lambda x: self.todense(cache=True) @ x

        super().__init__(
            shape=(self._linop.shape[1], self._linop.shape[0]),
            dtype=self._linop.dtype,
            matmul=matmul,
            todense=lambda: self._linop.todense(cache=False).T.copy(order="C"),
            transpose=lambda: self._linop,
        )

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "LinearOperator":
        return self._linop.astype(dtype, order=order, casting=casting, copy=copy).T

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
            matmul = lambda x: self.todense(cache=True) @ x

        super().__init__(
            shape=(self._linop.shape[1], self._linop.shape[0]),
            dtype=self._linop.dtype,
            matmul=matmul,
            todense=lambda: (
                np.conj(self._linop.todense(cache=False).T).copy(order="C")
            ),
            adjoint=lambda: self._linop,
        )

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "LinearOperator":
        return self._linop.astype(dtype, order=order, casting=casting, copy=copy).H


class _InverseLinearOperator(LinearOperator):
    def __init__(self, linop: LinearOperator):
        if not linop.is_square:
            raise np.linalg.LinAlgError("Only square operators can be inverted.")

        self._linop = linop

        super().__init__(
            shape=self._linop.shape,
            dtype=self._linop._inexact_dtype,
            matmul=lambda x: self.todense() @ x,
            todense=lambda: np.linalg.inv(self._linop.todense(cache=False)),
            inverse=lambda: self._linop,
        )


class _TypeCastLinearOperator(LinearOperator):
    def __init__(
        self,
        linop: LinearOperator,
        dtype: DTypeArgType,
        order: str = "K",
        casting: str = "unsafe",
        copy: bool = True,
    ):
        self._linop = linop

        dtype = np.dtype(dtype)

        if not np.can_cast(self._linop.dtype, dtype, casting=casting):
            raise TypeError(
                f"Cannot cast linear operator from {self._linop.dtype} to {dtype} "
                f"according to the rule {casting}"
            )

        super().__init__(
            self._linop.shape,
            dtype,
            matmul=lambda x: (self._linop @ x).astype(
                np.result_type(self.dtype, x.dtype), copy=False
            ),
            rmatmul=lambda x: (x @ self._linop).astype(
                np.result_type(x.dtype, self.dtype), copy=False
            ),
            apply=lambda x, axis: self._linop(x, axis=axis).astype(
                np.result_type(self.dtype, x.dtype), copy=False
            ),
            todense=lambda: self._linop.todense(cache=False).astype(
                dtype, order=order, copy=copy
            ),
            transpose=lambda: self._linop.T.astype(dtype),
            adjoint=lambda: self._linop.H.astype(dtype),
            inverse=lambda: self._linop.inv().astype(self._inexact_dtype),
            rank=self._linop.rank,
            eigvals=lambda: self._linop.eigvals().astype(self._inexact_dtype),
            cond=lambda p: self._linop.cond(p=p).astype(self._inexact_dtype),
            det=lambda: self._linop.det().astype(self._inexact_dtype),
            logabsdet=lambda: self._linop.logabsdet().astype(self._inexact_dtype),
            trace=lambda: self._linop.trace().astype(dtype),
        )

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "LinearOperator":
        if self.dtype == dtype and not copy:
            return self
        elif dtype == self._linop.dtype and not copy:
            return self._linop
        else:
            return _TypeCastLinearOperator(self, dtype, order, casting, copy)


class Matrix(LinearOperator):
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

        if isinstance(A, np.matrix):
            A = np.asarray(A)

        self.A = A

        if isinstance(self.A, scipy.sparse.spmatrix):
            matmul = LinearOperator.broadcast_matmat(lambda x: self.A @ x)
            rmatmul = LinearOperator.broadcast_rmatmat(lambda x: x @ self.A)
            todense = self.A.toarray
            inverse = lambda: Matrix(scipy.sparse.linalg.inv(self.A))
            trace = lambda: self.A.diagonal().sum()
        elif isinstance(self.A, np.ndarray):
            matmul = lambda x: A @ x
            rmatmul = lambda x: x @ A
            todense = lambda: self.A
            inverse = lambda: Matrix(np.linalg.inv(self.A))
            trace = lambda: np.trace(self.A)
        else:
            raise TypeError(
                f"`A` must be a `np.ndarray` or `scipy.sparse.spmatrix`, but a "
                f"`{type(self.A)}` was given."
            )

        transpose = lambda: Matrix(self.A.T)

        if np.issubdtype(dtype, np.complexfloating):
            adjoint = lambda: Matrix(np.conj(self.A.T))
        else:
            adjoint = transpose

        super().__init__(
            shape,
            dtype,
            matmul=matmul,
            rmatmul=rmatmul,
            todense=todense,
            transpose=transpose,
            adjoint=adjoint,
            inverse=inverse,
            trace=trace,
        )

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "LinearOperator":
        if isinstance(self.A, np.ndarray):
            A_astype = self.A.astype(dtype, order=order, casting=casting, copy=copy)
        else:
            assert isinstance(self.A, scipy.sparse.spmatrix)

            A_astype = self.A.astype(dtype, casting=casting, copy=copy)

        if A_astype is self.A:
            return self

        return Matrix(A_astype)
