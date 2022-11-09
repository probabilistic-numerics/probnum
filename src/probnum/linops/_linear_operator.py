"""Finite-dimensional linear operators."""

from __future__ import annotations

import abc
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from probnum import config
from probnum.typing import DTypeLike, ScalarLike, ShapeLike
import probnum.utils

BinaryOperandType = Union[
    "LinearOperator", ScalarLike, np.ndarray, scipy.sparse.spmatrix
]

# pylint: disable="too-many-lines"


class LinearOperator(abc.ABC):
    r"""Composite base class for finite-dimensional linear operators.

    This class provides a way to define finite-dimensional linear operators without
    explicitly constructing a matrix representation. Instead it suffices to define a
    matrix-vector product, a shape, and a ``dtype``. This avoids unnecessary memory
    usage and can often be more convenient to derive.

    :class:`LinearOperator` instances can be multiplied, added and scaled. This
    happens lazily: the result of these operations is a new, composite
    :class:`LinearOperator`, that defers linear operations to the original operators and
    combines the results.

    Parameters
    ----------
    shape
        Matrix dimensions `(M, N)`.
    dtype
        Data type of the operator.

    See Also
    --------
    aslinop : Transform into a LinearOperator.
    """

    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
    ):
        self.__shape = probnum.utils.as_shape(shape, ndim=2)

        # DType
        self.__dtype = np.dtype(dtype)

        if not np.issubdtype(self.__dtype, np.number):
            raise TypeError("The dtype of a linear operator must be numeric.")

        if np.issubdtype(self.__dtype, np.complexfloating):
            raise TypeError("Linear operators do not support complex dtypes.")

        # Matrix properties
        self._is_symmetric = None
        self._is_lower_triangular = None
        self._is_upper_triangular = None

        self._is_positive_definite = None

        # Caches
        self._todense_cache = None

        self._rank_cache = None
        self._eigvals_cache = None
        self._cond_cache = {}
        self._det_cache = None
        self._logabsdet_cache = None
        self._trace_cache = None

        self._lu_cache = None
        self._cholesky_cache = None

        # Property inference
        if not self.is_square:
            self.is_symmetric = False

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the linear operator.

        Defined as a tuple of the output and input dimension of operator.
        """
        return self.__shape

    @property
    def ndim(self) -> int:
        """Number of linear operator dimensions.

        Defined analogously to :attr:`numpy.ndarray.ndim`.
        """
        return 2

    @property
    def size(self) -> int:
        return self.__shape[0] * self.__shape[1]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the linear operator."""
        return self.__dtype

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def _apply(self, x: np.ndarray, axis: int) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Apply the linear operator to an input array along a specified
        axis.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        axis : int
            Axis along which to apply the linear operator.
            Guaranteed to be positive and valid, i.e. ``axis`` is a valid
            index into the shape of ``x``, and ``x`` has the correct
            shape along ``axis``.

        Returns
        -------
        apply_result : np.ndarray
            Array resulting in the application of the linear operator
            to ``x`` along ``axis``.
        """
        if axis is not None and (axis < -x.ndim or axis >= x.ndim):
            raise np.AxisError(axis, ndim=x.ndim)

        if x.ndim == 1:
            return self @ x
        elif x.ndim > 1:
            if axis is None:
                axis = -1

            if axis < 0:
                axis += x.ndim

            if x.shape[axis] != self.__shape[1]:
                raise ValueError(
                    f"Dimension mismatch. Expected array with {self.__shape[1]} "
                    f"entries along axis {axis}, but got array with shape {x.shape}."
                )

            if axis == (x.ndim - 1):
                return (self @ x[..., np.newaxis])[..., 0]
            elif axis == (x.ndim - 2):
                return self @ x
            else:
                try:
                    return self._apply(x, axis)
                except NotImplementedError:
                    return np.moveaxis(self @ np.moveaxis(x, axis, -2), -2, axis)
        else:
            raise ValueError("The operand must be at least one dimensional.")

    def astype(
        self,
        dtype: DTypeLike,
        order: str = "K",
        casting: str = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> "LinearOperator":
        """Cast a linear operator to a different ``dtype``.

        Parameters
        ----------
        dtype:
            Data type to which the linear operator is cast.
        order:
            Memory layout order of the result.
        casting:
            Controls what kind of data casting may occur.
        subok:
            If True, then sub-classes will be passed-through (default).
            False is currently not supported for linear operators.
        copy:
            Whether to return a new linear operator, even if ``dtype`` is the same.
        """
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

    def _todense(self) -> np.ndarray:
        """Dense matrix representation of the linear operator.

        You may implement this method in a subclass.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the linear operator.
        """
        raise NotImplementedError()

    def todense(self, cache: bool = True) -> np.ndarray:
        """Dense matrix representation of the linear operator.

        This method can be computationally very costly depending on the shape of the
        linear operator. Use with caution.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the linear operator.
        """
        if self._todense_cache is None:
            try:
                dense = self._todense()
            except NotImplementedError:
                dense = self @ np.eye(self.shape[1], dtype=self.__dtype, order="F")

            if not cache:
                return dense

            self._todense_cache = dense

        return self._todense_cache

    ####################################################################################
    # Matrix Properties
    ####################################################################################

    @property
    def is_square(self) -> bool:
        """Whether input dimension matches output dimension."""
        return self.shape[0] == self.shape[1]

    @property
    def is_symmetric(self) -> Optional[bool]:
        """Whether the ``LinearOperator`` :math:`L` is symmetric, i.e. :math:`L = L^T`.

        If this is ``None``, it is unknown whether the operator is symmetric or not.
        Only square operators can be symmetric."""
        return self._is_symmetric

    @is_symmetric.setter
    def is_symmetric(self, value: Optional[bool]) -> None:
        if value is True and not self.is_square:
            raise ValueError("Only square operators can be symmetric.")

        self._set_property("symmetric", value)

    @property
    def is_lower_triangular(self) -> Optional[bool]:
        """Whether the ``LinearOperator`` represents a lower triangular matrix.

        If this is ``None``, it is unknown whether the matrix is lower triangular or
        not.
        """
        return self._is_lower_triangular

    @is_lower_triangular.setter
    def is_lower_triangular(self, value: Optional[bool]) -> None:
        self._set_property("lower_triangular", value)

    @property
    def is_upper_triangular(self) -> Optional[bool]:
        """Whether the ``LinearOperator`` represents an upper triangular matrix.

        If this is ``None``, it is unknown whether the matrix is upper triangular or
        not.
        """
        return self._is_upper_triangular

    @is_upper_triangular.setter
    def is_upper_triangular(self, value: Optional[bool]) -> None:
        self._set_property("upper_triangular", value)

    @property
    def is_positive_definite(self) -> Optional[bool]:
        """Whether the ``LinearOperator`` :math:`L \\in \\mathbb{R}^{n \\times n}` is
        (strictly) positive-definite, i.e. :math:`x^T L x > 0` for :math:`x \\in \

        \\mathbb{R}^n`.

        If this is ``None``, it is unknown whether the matrix is positive-definite or
        not. Only symmetric operators can be positive-definite.
        """
        return self._is_positive_definite

    @is_positive_definite.setter
    def is_positive_definite(self, value: Optional[bool]) -> None:
        if value is True and not self.is_symmetric:
            raise ValueError("Only symmetric operators can be positive-definite.")

        self._set_property("positive_definite", value)

    def _set_property(self, name: str, value: Optional[bool]):
        attr_name = f"_is_{name}"

        try:
            curr_value = getattr(self, attr_name)
        except AttributeError as err:
            raise AttributeError(
                f"The matrix property `{name}` does not exist."
            ) from err

        if curr_value == value:
            return

        if curr_value is not None:
            assert isinstance(curr_value, bool)

            raise ValueError(f"Can not change the value of the matrix property {name}.")

        if not isinstance(value, bool):
            raise TypeError(
                f"The value of the matrix property {name} must be a boolean or "
                f"`None`, not {type(value)}."
            )

        setattr(self, attr_name, value)

    ####################################################################################
    # Derived Quantities
    ####################################################################################

    def _rank(self) -> np.intp:
        """Rank of the linear operator.

        You may implement this method in a subclass.
        """
        raise NotImplementedError()

    def rank(self) -> np.intp:
        """Rank of the linear operator."""
        if self._rank_cache is None:
            try:
                self._rank_cache = self._rank()
            except NotImplementedError:
                self._rank_cache = np.linalg.matrix_rank(self.todense(cache=False))

        return self._rank_cache

    def _eigvals(self) -> np.ndarray:
        """Eigenvalue spectrum of the linear operator.

        You may implement this method in a subclass.
        """
        raise NotImplementedError()

    def eigvals(self) -> np.ndarray:
        """Eigenvalue spectrum of the linear operator."""
        if self._eigvals_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "Eigenvalues are only defined for square operators"
                )

            try:
                self._eigvals_cache = self._eigvals()
            except NotImplementedError:
                self._eigvals_cache = np.linalg.eigvals(self.todense(cache=False))

            self._eigvals_cache.setflags(write=False)

        return self._eigvals_cache

    def _cond(
        self, p: Optional[Union[None, int, str, np.floating]] = None
    ) -> np.number:
        """Compute the condition number of the linear operator.

        The condition number of the linear operator with respect to the ``p`` norm. It
        measures how much the solution :math:`x` of the linear system :math:`Ax=b`
        changes with respect to small changes in :math:`b`.

        The linear operator is guaranteed to be square.

        You may implement this method in a subclass.

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
        cond :
            The condition number of the linear operator. May be infinite.
        """
        raise NotImplementedError()

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
        cond :
            The condition number of the linear operator. May be infinite.

        Raises
        ------
        LinAlgError :
            If :meth:`cond` is called on a non-square matrix.
        """
        if p not in self._cond_cache:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The condition number is only defined for square operators"
                )

            try:
                self._cond_cache[p] = self._cond(p)
            except NotImplementedError:
                self._cond_cache[p] = np.linalg.cond(self.todense(cache=False), p=p)

        return self._cond_cache[p]

    def _det(self) -> np.number:
        """Determinant of the linear operator.

        The linear operator is guaranteed to be square.

        You may implement this method in a subclass.

        Returns
        -------
        det :
            The determinant of the linear operator.
        """
        raise NotImplementedError()

    def det(self) -> np.inexact:
        """Determinant of the linear operator.

        Returns
        -------
        det :
            The determinant of the linear operator.

        Raises
        ------
        LinAlgError :
            If :meth:`det` is called on a non-square matrix.
        """
        if self._det_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The determinant is only defined for square operators"
                )

            try:
                self._det_cache = self._det()
            except NotImplementedError:
                self._det_cache = np.linalg.det(self.todense(cache=False))

        return self._det_cache

    def _logabsdet(self) -> np.flexible:
        """Log absolute determinant of the linear operator.

        The linear operator is guaranteed to be square.

        You may implement this method in a subclass.

        Returns
        -------
        logabsdet :
            The log absolute determinant of the linear operator.
        """
        raise NotImplementedError()

    def logabsdet(self) -> np.inexact:
        """Log absolute determinant of the linear operator.

        Returns
        -------
        logabsdet :
            The log absolute determinant of the linear operator.

        Raises
        ------
        LinAlgError :
            If :meth:`logabsdet` is called on a non-square matrix.
        """
        if self._logabsdet_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The determinant is only defined for square operators"
                )

            try:
                self._logabsdet_cache = self._logabsdet()
            except NotImplementedError:
                self._logabsdet_cache = self._logabsdet_fallback()

        return self._logabsdet_cache

    def _logabsdet_fallback(self) -> np.inexact:
        if self.det() == 0:
            return probnum.utils.as_numpy_scalar(-np.inf, dtype=self._inexact_dtype)
        else:
            return np.log(np.abs(self.det()))

    def _trace(self) -> np.number:
        r"""Trace of the linear operator.

        Computes the trace of a square linear operator :math:`\text{tr}(A) =
        \sum_{i-1}^n A_{ii}`.

        The linear operator is guaranteed to be square.

        You may implement this method in a subclass.

        Returns
        -------
        trace : float
            Trace of the linear operator.
        """
        raise NotImplementedError()

    def trace(self) -> np.number:
        r"""Trace of the linear operator.

        Computes the trace of a square linear operator :math:`\text{tr}(A) =
        \sum_{i-1}^n A_{ii}`.

        Returns
        -------
        trace : float
            Trace of the linear operator.

        Raises
        ------
        LinAlgError :
            If :meth:`trace` is called on a non-square matrix.
        """
        if self._trace_cache is None:
            if not self.is_square:
                raise np.linalg.LinAlgError(
                    "The trace is only defined for square operators."
                )

            try:
                self._trace_cache = self._trace()
            except NotImplementedError:
                self._trace_cache = self._trace_fallback()

        return self._trace_cache

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

    ####################################################################################
    # Matrix Decompositions
    ####################################################################################

    def cholesky(self, lower: bool = True) -> LinearOperator:
        r"""Computes a Cholesky decomposition of the :class:`LinearOperator`.

        The Cholesky decomposition of a symmetric positive-definite matrix :math:`A \in
        \mathbb{R}^{n \times n}` is given by :math:`A = L L^T`, where the unique
        Cholesky factor :math:`L \in \mathbb{R}^{n \times n}` of :math:`A` is a lower
        triangular matrix with a positive diagonal.

        As a side-effect, this method will set the value of :attr:`is_positive_definite`
        to :obj:`True`, if the computation of the Cholesky factorization succeeds.
        Otherwise, :attr:`is_positive_definite` will be set to :obj:`False`.

        The result of this computation will be cached. If :meth:`cholesky` is first
        called with ``lower=True`` first and afterwards with ``lower=False`` or
        vice-versa, the method simply returns the transpose of the cached value.

        Parameters
        ----------
        lower :
            If this is set to :obj:`False`, this method computes and returns the
            upper triangular Cholesky factor :math:`U = L^T`, for which :math:`A = U^T
            U`. By default (:obj:`True`), the method computes the lower triangular
            Cholesky factor :math:`L`.

        Returns
        -------
        cholesky_factor :
            The lower or upper Cholesky factor of the :class:`LinearOperator`, depending
            on the value of the parameter ``lower``. The result will have its properties
            :attr:`is_upper_triangular`\ /:attr:`is_lower_triangular` set accordingly.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the :class:`LinearOperator` is not symmetric, i.e. if
            :attr:`is_symmetric` is not set to :obj:`True`.
        numpy.linalg.LinAlgError
            If the :class:`LinearOperator` is not positive definite.
        """
        if not self.is_symmetric:
            raise np.linalg.LinAlgError(
                "The Cholesky decomposition is only defined for symmetric matrices."
            )

        if self.is_positive_definite is False:
            raise np.linalg.LinAlgError("The linear operator is not positive definite.")

        if self._cholesky_cache is None:
            try:
                self._cholesky_cache = self._cholesky(lower)

                self.is_positive_definite = True
            except np.linalg.LinAlgError as err:
                self.is_positive_definite = False

                raise err

            if lower:
                self._cholesky_cache.is_lower_triangular = True
            else:
                self._cholesky_cache.is_upper_triangular = True

        upper = not lower

        if (lower and self._cholesky_cache.is_lower_triangular) or (
            upper and self._cholesky_cache.is_upper_triangular
        ):
            return self._cholesky_cache

        assert (
            self._cholesky_cache.is_lower_triangular
            or self._cholesky_cache.is_upper_triangular
        )

        return self._cholesky_cache.T

    def _cholesky(self, lower: bool) -> LinearOperator:
        return Matrix(
            scipy.linalg.cholesky(
                self.todense(), lower=lower, overwrite_a=False, check_finite=True
            )
        )

    def _lu_factor(self):
        """This is a modified version of the original implementation in SciPy:

        https://github.com/scipy/scipy/blob/v1.7.1/scipy/linalg/decomp_lu.py#L15-L84
        because the SciPy implementation does not raise an exception if the matrix is
        singular.
        """

        if self._lu_cache is None:
            from scipy.linalg.lapack import (  # pylint: disable=no-name-in-module,import-outside-toplevel
                get_lapack_funcs,
            )

            a = np.asarray_chkfinite(self.todense())
            (getrf,) = get_lapack_funcs(("getrf",), (a,))
            lu, piv, info = getrf(a, overwrite_a=False)

            if info < 0:
                raise ValueError(
                    f"illegal value in argument {-info} of internal getrf (lu_factor)"
                )

            if info > 0:
                raise np.linalg.LinAlgError(
                    f"Diagonal number {info} is exactly zero. Singular matrix."
                )

            self._lu_cache = lu, piv

        return self._lu_cache

    ####################################################################################
    # Unary Arithmetic
    ####################################################################################

    def __neg__(self) -> "LinearOperator":
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            NegatedLinearOperator,
        )

        return NegatedLinearOperator(self)

    def _transpose(self) -> "LinearOperator":
        """Transpose of this linear operator.

        You may implement this method in a subclass.

        Returns
        -------
        transpose : LinearOperator
            Transpose of this linear operator, which is again
            a LinearOperator.
        """
        raise NotImplementedError()

    @property
    def T(self) -> "LinearOperator":
        try:
            return self._transpose()
        except NotImplementedError:
            pass

        # This does not need caching, since the `TransposeLinearOperator` only accesses
        # quantities (particularly `todense`), which are cached inside the original
        # `LinearOperator`.
        return TransposedLinearOperator(self)

    def transpose(self, *axes: Union[int, Tuple[int]]) -> "LinearOperator":
        """Transpose this linear operator.

        Can be abbreviated self.T instead of self.transpose().
        """
        if len(axes) > 0:
            if len(axes) == 1 and isinstance(axes[0], tuple):
                axes = axes[0]

            if len(axes) != 2:
                raise ValueError(
                    f"The given axes {axes} don't match the linear operator with shape "
                    f"{self.shape}."
                )

            axes_int = []

            for axis in axes:
                axis = int(axis)

                if not -2 <= axis <= 1:
                    raise np.AxisError(axis, ndim=2)

                if axis < 0:
                    axis += 2

                axes_int.append(axis)

            axes = tuple(axes_int)

            if axes == (0, 1):
                return self

            if axes == (1, 0):
                return self.T

            raise ValueError("Cannot transpose a linear operator along repeated axes.")

        return self.T

    def _inverse(self) -> "LinearOperator":
        """Inverse of this linear operator.

        You may implement this method in a subclass.

        Returns
        -------
        inv : LinearOperator
            Inverse of this linear operator, which is again
            a LinearOperator.
        """
        raise NotImplementedError()

    def inv(self) -> "LinearOperator":
        """Inverse of the linear operator.

        Returns
        -------
        inv : LinearOperator
            Inverse of this linear operator, which is again
            a LinearOperator.
        """
        try:
            return self._inverse()
        except NotImplementedError:
            pass

        # This does not need caching, since the `_InverseLinearOperator` only accesses
        # quantities (particularly matrix decompositions), which are cached inside the
        # original `LinearOperator`.
        return _InverseLinearOperator(self)

    def symmetrize(self) -> LinearOperator:
        """Compute or approximate the closest symmetric :class:`LinearOperator` in the
        Frobenius norm.

        The closest symmetric matrix to a given square matrix :math:`A` in the Frobenius
        norm is given by

        .. math::
            \\operatorname{sym}(A) := \\frac{1}{2} (A + A^T).

        However, for efficiency reasons, it is preferrable to approximate this operator
        in some cases. For example, a Kronecker product :math:`K = A \\otimes B` is more
        efficiently symmetrized by means of

        .. math::
            :nowrap:

            \\begin{equation}
                \\operatorname{sym}(A) \\otimes \\operatorname{sym}(B)
                = \\operatorname{sym}(K) + \\frac{1}{2} \\left(
                    \\frac{1}{2} \\left(
                        A \\otimes B^T + A^T \\otimes B
                    \\right) - \\operatorname{sym}(K)
                \\right).
            \\end{equation}

        Returns
        -------
        symmetrized_linop :
            The closest symmetric :class:`LinearOperator` in the Frobenius norm, or an
            approximation, which makes a reasonable trade-off between accuracy and
            efficiency (see above). The resulting :class:`LinearOperator` will have its
            :attr:`is_symmetric` property set to :obj:`True`.

        Raises
        ------
        numpy.linalg.LinAlgError
            If this method is called on a non-square :class:`LinearOperator`.
        """
        if not self.is_square:
            raise np.linalg.LinAlgError("A non-square operator can not be symmetrized.")

        if self.is_symmetric:
            return self

        linop_sym = self._symmetrize()
        linop_sym.is_symmetric = True

        return linop_sym

    def _symmetrize(self) -> LinearOperator:
        return 0.5 * (self + self.T)

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

    def _is_type_shape_dtype_equal(self, other: "LinearOperator") -> bool:
        return (
            isinstance(self, type(other))
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    @abc.abstractmethod
    def _matmul(self, x: np.ndarray) -> np.ndarray:
        """Matrix multiplication.

        Performs the operation `M = self @ x` where `self` is
        an MxN linear operator and `x` is a stack of matrices
        of shape `(..., N, K)`.

        The shapes are guaranteed to be correct.

        You must implement this method in a subclass.

        Parameters
        ----------
        x :
            A stack of matrices of shape `(..., N, K)`.

        Returns
        -------
        M :
            A `np.ndarray` of shape `(..., M, K)` that is the result of
            `M = self @ x`.
        """
        raise NotImplementedError()

    def __matmul__(
        self, other: BinaryOperandType
    ) -> Union["LinearOperator", np.ndarray]:
        """Matrix-vector multiplication.

        Performs the operation `y = self @ x` where `self` is
        an MxN linear operator and `x` is a 1-d array or random variable.

        Parameters
        ----------
        x :
            An array or `RandomVariable` with shape `(N,)` or `(N, 1)`.

        Returns
        -------
        y :
            A `np.matrix` or `np.ndarray` or `RandomVariable` with
            shape `(M,)` or `(M, 1)`,depending on the type and
            shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden
        _matvec method to ensure that y has the correct shape and type.
        """

        if isinstance(other, np.ndarray):
            x = other

            M, N = self.shape

            if x.ndim == 1 and x.shape == (N,):
                y = self._matmul(x[:, np.newaxis])[:, 0]

                assert y.ndim == 1
                assert y.shape == (M,)
            elif x.ndim > 1 and x.shape[-2] == N:
                y = self._matmul(x)

                assert y.ndim > 1
                assert y.shape == x.shape[:-2] + (M, x.shape[-1])
            else:
                raise ValueError(
                    f"Dimension mismatch. Expected operand of shape ({N},) or "
                    f"(..., {N}, K), but got {x.shape}."
                )

            return y
        else:
            from ._arithmetic import matmul  # pylint: disable=import-outside-toplevel

            return matmul(self, other)

    def __rmatmul__(
        self, other: BinaryOperandType
    ) -> Union["LinearOperator", np.ndarray]:
        if isinstance(other, np.ndarray):
            x = other

            M, N = self.shape

            if x.ndim >= 1 and x.shape[-1] == M:
                y = (self.T)(x, axis=-1)  # pylint: disable=not-callable
            else:
                raise ValueError(
                    f"Dimension mismatch. Expected operand of shape (..., {M}), but "
                    f"got {x.shape}."
                )

            assert y.ndim >= 1
            assert y.shape[-1] == N
            assert y.shape[:-1] == x.shape[:-1]

            return y
        else:
            from ._arithmetic import matmul  # pylint: disable=import-outside-toplevel

            return matmul(other, self)

    ####################################################################################
    # Automatic `mat{vec,mat}`` to `matmul` Broadcasting
    ####################################################################################

    @classmethod
    def broadcast_matvec(
        cls, matvec: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Broadcasting for a (implicitly defined) matrix-vector product.

        Convenience function / decorator to broadcast the definition of a matrix-vector
        product. This can be used to easily construct a new linear operator only from a
        matrix-vector product.
        """

        def _matmul(x: np.ndarray) -> np.ndarray:
            if x.ndim == 2 and x.shape[1] == 1:
                return matvec(x[:, 0])[:, np.newaxis]

            return np.apply_along_axis(matvec, -2, x)

        return _matmul

    @classmethod
    def broadcast_matmat(
        cls, matmat: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Broadcasting for a (implicitly defined) matrix-matrix product.

        Convenience function / decorator to broadcast the definition of a matrix-matrix
        product to stacks of matrices. This can be used to easily construct a new linear
        operator only from a matrix-matrix product.
        """
        return np.vectorize(matmat, signature="(n,k)->(m,k)")

    @property
    def _inexact_dtype(self) -> np.dtype:
        if np.issubdtype(self.dtype, np.inexact):
            return self.dtype
        else:
            return np.double


def _call_if_implemented(method: Optional[callable]) -> callable:
    if method is not None:
        return method
    raise NotImplementedError()


class LambdaLinearOperator(LinearOperator):
    r"""Convenience subclass of LinearOperator that lets you pass
    implementations of its methods as parameters instead of
    overriding them in a subclass.

    ``shape``, ``dtype`` and ``matmul`` must be passed, the other
    parameters are optional.

    Parameters
    ----------
    shape
        Matrix dimensions `(M, N)`.
    dtype
        Data type of the operator.
    matmul
        Callable which computes the matrix-matrix product :math:`y = A V`, where
        :math:`A` is the linear operator and :math:`V` is an :math:`N \times K` matrix.
        The callable must support broadcasted matrix products, i.e. the argument
        :math:`V` might also be a stack of matrices in which case the broadcasting rules
        of :func:`np.matmul` must apply.
        Note that the argument to this callable is guaranteed to have at least two
        dimensions.
    rmatmul
        Callable which implements the matrix-matrix product, i.e. :math:`A @ V`, where
        :math:`A` is the linear operator and :math:`V` is a matrix of shape `(N, K)`.
    apply
        Callable which implements the application of the linear operator to an input
        array along a specified axis.
    todense
        Callable which returns a dense matrix representation of the linear operator as a
        :class:`np.ndarray`. The output of this function must be equivalent to the
        output of :code:`A.matmat(np.eye(N, dtype=A.dtype))`.
    transpose
        Callable which returns a LinearOperator that corresponds to the
        transpose of this linear operator.
    inverse
        Callable which returns a LinearOperator that corresponds to the
        inverse of this linear operator.
    rank
        Callable which returns the rank of this linear operator.
    eigvals
        Callable which returns the eigenvalues of this linear operator.
    cond
        Callable which returns the condition number of this
        linear operator.
    det
        Callable which returns the determinant of this linear operator.
    logabsdet
        Callable which returns the log absolute determinant of this
        linear operator.
    trace
        Callable which returns the trace of this linear operator.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.linops import LambdaLinearOperator, LinearOperator

    >>> @LinearOperator.broadcast_matvec
    ... def mv(v):
    ...     return np.array([2 * v[0] - v[1], 3 * v[1]])

    >>> A = LambdaLinearOperator(shape=(2, 2), dtype=np.float_, matmul=mv)
    >>> A
    <LambdaLinearOperator with shape=(2, 2) and dtype=float64>

    >>> A @ np.array([1., 2.])
    array([0., 6.])
    >>> A @ np.ones(2)
    array([1., 3.])
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike,
        *,
        matmul: Callable[[np.ndarray], np.ndarray],
        rmatmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        apply: Callable[[np.ndarray, int], np.ndarray] = None,
        todense: Optional[Callable[[], np.ndarray]] = None,
        transpose: Optional[Callable[[np.ndarray], "LinearOperator"]] = None,
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
        super().__init__(shape, dtype)

        self._matmul_fn = matmul  # (self @ x)
        self._rmatmul_fn = rmatmul  # (x @ self)
        self._apply_fn = apply  # __call__

        self._todense_fn = todense

        self._transpose_fn = transpose
        self._inverse_fn = inverse

        # Derived quantities
        self._rank_fn = rank
        self._eigvals_fn = eigvals
        self._cond_fn = cond
        self._det_fn = det
        self._logabsdet_fn = logabsdet
        self._trace_fn = trace

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        return self._matmul_fn(x)

    def _apply(self, x: np.ndarray, axis: int) -> np.ndarray:
        return _call_if_implemented(self._apply_fn)(x, axis)

    def _todense(self) -> np.ndarray:
        return _call_if_implemented(self._todense_fn)()

    def _transpose(self) -> "LinearOperator":
        if self._transpose_fn is not None:
            return self._transpose_fn()

        if self._rmatmul_fn is not None:
            return TransposedLinearOperator(
                self,
                matmul=lambda x: np.moveaxis(
                    self._rmatmul_fn(np.moveaxis(x, -2, -1)), -1, -2
                ),
            )

        raise NotImplementedError()

    def _inverse(self) -> "LinearOperator":
        return _call_if_implemented(self._inverse_fn)()

    def _rank(self) -> np.intp:
        return _call_if_implemented(self._rank_fn)()

    def _eigvals(self) -> np.ndarray:
        return _call_if_implemented(self._eigvals_fn)()

    def _cond(
        self, p: Optional[Union[None, int, str, np.floating]] = None
    ) -> np.number:
        return _call_if_implemented(self._cond_fn)(p)

    def _det(self) -> np.number:
        return _call_if_implemented(self._det_fn)()

    def _logabsdet(self) -> np.flexible:
        return _call_if_implemented(self._logabsdet_fn)()

    def _trace(self) -> np.number:
        return _call_if_implemented(self._trace_fn)()


class TransposedLinearOperator(LambdaLinearOperator):
    """Transposition of a linear operator."""

    def __init__(
        self,
        linop: LinearOperator,
        matmul: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self._linop = linop

        if matmul is None:
            # Setting `cache=True` here does not allocate any extra memory, since the
            # transpose of `self._linop.todense(cache=True)` is just a view of the
            # original array
            matmul = lambda x: self.todense(cache=True) @ x

        super().__init__(
            shape=(self._linop.shape[1], self._linop.shape[0]),
            dtype=self._linop.dtype,
            matmul=matmul,
            rmatmul=lambda x: self._linop(x, axis=-1),
            todense=lambda: self._linop.todense(cache=True).T,
            transpose=lambda: self._linop,
            inverse=None,  # lambda: self._linop.inv().T,
            rank=self._linop.rank,
            det=self._linop.det,
            logabsdet=self._linop.logabsdet,
            trace=self._linop.trace,
        )

        # Property Inference
        self.is_symmetric = self._linop.is_symmetric
        self.is_positive_definite = self._linop.is_positive_definite

        if self._linop.is_lower_triangular:
            self._linop.is_upper_triangular = True

        if self._linop.is_upper_triangular:
            self._linop.is_lower_triangular = True

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "LinearOperator":
        return self._linop.astype(dtype, order=order, casting=casting, copy=copy).T

    def __repr__(self) -> str:
        return f"Transpose of {self._linop}"

    def _cholesky(self, lower: bool = True) -> LinearOperator:
        return super().cholesky(lower)


class _InverseLinearOperator(LambdaLinearOperator):
    def __init__(self, linop: LinearOperator):
        if not linop.is_square:
            raise np.linalg.LinAlgError("Only square operators can be inverted.")

        self._linop = linop

        solve = np.vectorize(
            self._solve,
            excluded=("trans",),
            signature="(n, k)->(n, k)",
        )

        super().__init__(
            shape=self._linop.shape,
            dtype=self._linop._inexact_dtype,
            matmul=solve,
            transpose=lambda: TransposedLinearOperator(
                self,
                matmul=lambda x: solve(x, trans=True),
            ),
            inverse=lambda: self._linop,
            det=lambda: 1 / self._linop.det(),
            logabsdet=lambda: -self._linop.logabsdet(),
        )

        # Matrix properties
        self.is_symmetric = self._linop.is_symmetric
        self.is_positive_definite = self._linop.is_positive_definite

    def __repr__(self) -> str:
        return f"Inverse of {self._linop}"

    def _solve(self, x: np.ndarray, trans: bool = False) -> np.ndarray:
        """Solve :math:`A Y = X` for Y, where either :code:`A = self._linop` or
        :code:`A = self._linop.T`, depending on the value of :code:`trans`.

        Parameters
        ----------
        x
            :code:`shape=(N,K)` --
            The right-hand sides :math:`X` of the linear systems, where
            :code:`A.shape == (N, N)`.
        trans
            If :code:`False`, then :code:`A = self._linop`.
            Otherwise :code:`A = self._linop.T`.

        Returns
        -------
        sol
            The solutions :math:`A^{-1} X` of the linear systems.
        """
        assert x.ndim == 2

        if self._linop.is_symmetric:
            if self._linop.is_positive_definite is not False:
                try:
                    # A @ x = A^T @ x, since A is symmetric
                    return scipy.linalg.cho_solve(
                        (self._linop.cholesky(lower=False).todense(), False),
                        x,
                        overwrite_b=False,
                    )
                except np.linalg.LinAlgError:
                    pass

        return scipy.linalg.lu_solve(
            self._linop._lu_factor(),
            x,
            trans=1 if trans else 0,
            overwrite_b=False,
        )


class _TypeCastLinearOperator(LambdaLinearOperator):
    def __init__(
        self,
        linop: LinearOperator,
        dtype: DTypeLike,
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


class Matrix(LambdaLinearOperator):
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
        if isinstance(A, scipy.sparse.spmatrix):
            self.A = A

            shape = self.A.shape
            dtype = self.A.dtype

            matmul = LinearOperator.broadcast_matmat(lambda x: self.A @ x)
            todense = self.A.toarray
            inverse = self._sparse_inv
            trace = lambda: self.A.diagonal().sum()
        else:
            self.A = np.asarray(A)

            shape = self.A.shape
            dtype = self.A.dtype

            matmul = lambda x: self.A @ x
            todense = lambda: self.A
            inverse = None
            trace = lambda: np.trace(self.A)

        transpose = lambda: Matrix(self.A.T)

        super().__init__(
            shape,
            dtype,
            matmul=matmul,
            todense=todense,
            transpose=transpose,
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

    def _sparse_inv(self) -> "Matrix":
        try:
            return Matrix(scipy.sparse.linalg.inv(self.A.tocsc()))
        except RuntimeError as err:
            raise np.linalg.LinAlgError(str(err)) from err

    def _matmul_matrix(self, other: "Matrix") -> "Matrix":
        if not config.lazy_matrix_matrix_matmul:
            if not self.shape[1] == other.shape[0]:
                raise ValueError(f"Matmul shape mismatch {self.shape} x {other.shape}")

            return Matrix(A=self.A @ other.A)

        return NotImplemented

    def __eq__(self, other: LinearOperator) -> bool:
        if not self._is_type_shape_dtype_equal(other):
            return False

        return np.all(self.A == other.A)

    def __neg__(self) -> "Matrix":
        return Matrix(-self.A)

    def _symmetrize(self) -> LinearOperator:
        return Matrix(0.5 * (self.A + self.A.T))


class Identity(LambdaLinearOperator):
    """The identity operator.

    Parameters
    ----------
    shape :
        Shape of the identity operator.
    """

    def __init__(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = np.double,
    ):
        shape = probnum.utils.as_shape(shape)

        if len(shape) == 1:
            shape = 2 * shape
        elif len(shape) != 2:
            raise ValueError("The shape of a linear operator must be two-dimensional.")

        if shape[0] != shape[1]:
            raise np.linalg.LinAlgError("An identity operator must be square.")

        super().__init__(
            shape,
            dtype,
            matmul=lambda x: x.astype(np.result_type(self.dtype, x.dtype), copy=False),
            rmatmul=lambda x: x.astype(np.result_type(self.dtype, x.dtype), copy=False),
            apply=lambda x, axis: x.astype(
                np.result_type(self.dtype, x.dtype), copy=False
            ),
            todense=lambda: np.identity(self.shape[0], dtype=dtype),
            transpose=lambda: self,
            inverse=lambda: self,
            rank=lambda: np.intp(shape[0]),
            eigvals=lambda: np.ones(shape[0], dtype=self._inexact_dtype),
            cond=self._cond,
            det=lambda: probnum.utils.as_numpy_scalar(1.0, dtype=self._inexact_dtype),
            logabsdet=lambda: probnum.utils.as_numpy_scalar(
                0.0, dtype=self._inexact_dtype
            ),
            trace=lambda: probnum.utils.as_numpy_scalar(
                self.shape[0], dtype=self.dtype
            ),
        )

        # Matrix properties
        self.is_symmetric = True
        self.is_lower_triangular = True
        self.is_upper_triangular = True

        self.is_positive_definite = True

    def _cond(
        self, p: Optional[Union[None, int, str, np.floating]] = None
    ) -> np.inexact:
        if p is None or p in (2, 1, np.inf, -2, -1, -np.inf):
            return probnum.utils.as_numpy_scalar(1.0, dtype=self._inexact_dtype)
        elif p == "fro":
            return probnum.utils.as_numpy_scalar(
                self.shape[0], dtype=self._inexact_dtype
            )
        else:
            return np.linalg.cond(self.todense(cache=False), p=p)

    def _astype(
        self, dtype: np.dtype, order: str, casting: str, copy: bool
    ) -> "Identity":
        if dtype == self.dtype and not copy:
            return self
        else:
            return Identity(self.shape, dtype=dtype)

    def __eq__(self, other: LinearOperator) -> bool:
        return self._is_type_shape_dtype_equal(other)

    def _cholesky(self, lower: bool = True) -> LinearOperator:
        return self


class Selection(LambdaLinearOperator):
    def __init__(self, indices, shape, dtype=np.double):
        if np.ndim(indices) > 1:
            raise ValueError(
                "Selection LinOp expects an integer or (1D) iterable of "
                f"integers. Received {type(indices)} with shape {np.shape(indices)}."
            )
        if shape[0] > shape[1]:
            raise ValueError(
                f"Invalid shape {shape} for Selection LinOp. If the "
                "output-dimension (shape[0]) is larger than the input-dimension "
                "(shape[1]), consider using `Embedding`."
            )
        self._indices = probnum.utils.as_shape(indices)
        assert len(self._indices) == shape[0]

        super().__init__(
            dtype=dtype,
            shape=shape,
            matmul=lambda x: _selection_matmul(self.indices, x),
            todense=self._todense,
            transpose=lambda: Embedding(
                take_indices=np.arange(len(self._indices)),
                put_indices=self._indices,
                shape=(self.shape[1], self.shape[0]),
            ),
        )

    @property
    def indices(self):
        return self._indices

    def _todense(self):
        res = np.eye(self.shape[1], self.shape[1])
        return _selection_matmul(self.indices, res)


def _selection_matmul(indices, M):
    return np.take(M, indices=indices, axis=-2)


class Embedding(LambdaLinearOperator):
    def __init__(
        self, take_indices, put_indices, shape, fill_value=0.0, dtype=np.double
    ):
        if np.ndim(take_indices) > 1:
            raise ValueError(
                "Embedding LinOp expects an integer or (1D) iterable of "
                f"integers. Received {type(take_indices)} with shape "
                f"{np.shape(take_indices)}."
            )
        if np.ndim(put_indices) > 1:
            raise ValueError(
                "Embedding LinOp expects an integer or (1D) iterable of "
                f"integers. Received {type(put_indices)} with shape "
                f"{np.shape(put_indices)}."
            )

        if shape[0] < shape[1]:
            raise ValueError(
                f"Invalid shape {shape} for Embedding LinOp. If the "
                "output-dimension (shape[0]) is smaller than the input-dimension "
                "(shape[1]), consider using `Selection`."
            )

        self._take_indices = probnum.utils.as_shape(take_indices)
        self._put_indices = probnum.utils.as_shape(put_indices)
        self._fill_value = fill_value

        super().__init__(
            dtype=dtype,
            shape=shape,
            matmul=lambda x: _embedding_matmul(self, x),
            todense=self._todense,
            transpose=lambda: Selection(
                indices=put_indices, shape=(self.shape[1], self.shape[0])
            ),
        )

    def _todense(self):
        return self.T.todense().T


def _embedding_matmul(embedding, M):
    res_shape = np.array(M.shape)
    res_shape[-2] = embedding.shape[0]
    res = np.full(shape=tuple(res_shape), fill_value=embedding._fill_value)
    take_from_M = M[..., np.array(embedding._take_indices), :]
    res[..., np.array(embedding._put_indices), :] = take_from_M
    return res
