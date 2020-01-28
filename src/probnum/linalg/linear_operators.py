"""Finite dimensional linear operators.

This module defines classes and methods that implement finite dimensional linear operators. It can be used to do linear
algebra with (structured) matrices without explicitly representing them in memory. This often allows for the definition
of a more efficient matrix-vector product. Linear operators can be applied, added, multiplied, transposed, and more as
one would expect from matrix algebra.

Several algorithms in the :mod:`probnum.linalg` library are able to operate on :class:`LinearOperator` instances.
"""
import warnings

import numpy as np
import scipy.sparse.linalg
import scipy.sparse.linalg.interface

__all__ = ["LinearOperator", "MatrixMult", "Identity", "Diagonal", "Kronecker", "SymmetricKronecker", "aslinop"]


class LinearOperator(scipy.sparse.linalg.LinearOperator):
    """
    Finite dimensional linear operators.

    This class provides a way to define finite dimensional linear operators without explicitly constructing a matrix
    representation. Instead it suffices to define a matrix-vector product and a shape attribute. This avoids unnecessary
    memory usage and can often be more convenient to derive.

    LinearOperator instances can be multiplied, added and exponentiated. This happens lazily: the result of these
    operations is a new, composite LinearOperator, that defers linear operations to the original operators and combines
    the results.

    To construct a concrete LinearOperator, either pass appropriate callables to the constructor of this class, or
    subclass it.

    A subclass must implement either one of the methods ``_matvec`` and ``_matmat``, and the
    attributes/properties ``shape`` (pair of integers) and ``dtype`` (may be ``None``). It may call the ``__init__`` on
    this class to have these attributes validated. Implementing ``_matvec`` automatically implements ``_matmat`` (using
    a naive algorithm) and vice-versa.

    Optionally, a subclass may implement ``_rmatvec`` or ``_adjoint`` to implement the Hermitian adjoint (conjugate
    transpose). As with ``_matvec`` and ``_matmat``, implementing either ``_rmatvec`` or ``_adjoint`` implements the
    other automatically. Implementing ``_adjoint`` is preferable; ``_rmatvec`` is mostly there for backwards
    compatibility.

    This class inherits from :class:`scipy.sparse.linalg.LinearOperator`.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M, N).
    matvec : callable f(v)
        Returns :math:`A v`.
    rmatvec : callable f(v)
        Returns :math:`A^H v`, where :math:`A^H` is the conjugate transpose of :math:`A`.
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
    >>> from probnum.linalg.linear_operators import LinearOperator
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

            if (type(obj)._matvec == scipy.sparse.linalg.LinearOperator._matvec
                    and type(obj)._matmat == scipy.sparse.linalg.LinearOperator._matmat):
                warnings.warn("LinearOperator subclass should implement"
                              " at least one of _matvec and _matmat.",
                              category=RuntimeWarning, stacklevel=2)

            return obj

    # Overload arithmetic operators to give access to newly implemented functions (e.g. todense())
    def __rmul__(self, x):
        if np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __pow__(self, p):
        if np.isscalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        else:
            return NotImplemented

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

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
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if len(x.shape) == 1 or len(x.shape) == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif len(x.shape) == 2:
                return self.matmat(x)
            else:
                raise ValueError('Expected 1-d or 2-d array, matrix or random variable, got %r.' % x)

    def matvec(self, x):
        """Matrix-vector multiplication.
        Performs the operation y=A*x where A is an MxN linear
        operator and x is a 1-d array or random variable.

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
            raise ValueError('Dimension mismatch.')

        y = self._matvec(x)

        if isinstance(x, np.matrix):
            y = scipy.sparse.sputils.asmatrix(y)

        if isinstance(x, (np.matrix, np.ndarray)):
            if x.ndim == 1:
                y = y.reshape(M)
            elif x.ndim == 2:
                y = y.reshape(M, 1)
            else:
                raise ValueError('Invalid shape returned by user-defined matvec().')
        # TODO: can be shortened once RandomVariable implements a reshape method
        elif y.shape[0] != M:
            raise ValueError('Invalid shape returned by user-defined matvec().')

        return y

    def transpose(self):
        """
        Transpose this linear operator.

        Can be abbreviated self.T instead of self.transpose().
        """
        return self._transpose()

    T = property(transpose)

    def _transpose(self):
        """ Default implementation of _transpose; defers to rmatvec + conj"""
        return _TransposedLinearOperator(self)

    def todense(self):
        """
        Dense matrix representation of the linear operator.

        This method can be computationally very costly depending on the shape of the linear operator. Use with caution.

        Returns
        -------
        matrix : np.ndarray
            Matrix representation of the linear operator.
        """
        return self.matmat(np.eye(self.shape[1], dtype=self.dtype))

    def rank(self):
        """Rank of the linear operator."""
        # TODO: implement operations (eigs, cond, det, logabsdet, trace, ...)
        raise NotImplementedError

    def eigvals(self):
        """Eigenvalue spectrum of the linear operator."""
        raise NotImplementedError

    def cond(self, p=None):
        """
        Compute the condition number of the linear operator.

        The condition number of the linear operator with respect to the ``p`` norm. It measures how much the solution
        :math:`x` of the linear system :math:`Ax=b` changes with respect to small changes in :math:`b`.

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
        """Trace of the linear operator."""
        raise NotImplementedError


class _CustomLinearOperator(scipy.sparse.linalg.interface._CustomLinearOperator, LinearOperator):
    """Linear operator defined in terms of user-specified operations."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None,
                 rmatmat=None, dtype=None):
        super().__init__(shape=shape, matvec=matvec, rmatvec=rmatvec, matmat=matmat, rmatmat=rmatmat, dtype=dtype)


# TODO: inheritance from _TransposedLinearOperator causes dependency on scipy>=1.4, maybe implement our own instead?
class _TransposedLinearOperator(scipy.sparse.linalg.interface._TransposedLinearOperator, LinearOperator):
    """Transposition of a linear operator."""

    def __init__(self, A):
        super().__init__(A=A)

    def todense(self):
        A = self.args[0]
        return A.todense().T


class _SumLinearOperator(scipy.sparse.linalg.interface._SumLinearOperator, LinearOperator):
    """Sum of two linear operators."""

    def __init__(self, A, B):
        super().__init__(A=A, B=B)

    def todense(self):
        A, B = self.args
        return A.todense() + B.todense()


class _ProductLinearOperator(scipy.sparse.linalg.interface._ProductLinearOperator, LinearOperator):
    """(Operator) Product of two linear operators."""

    def __init__(self, A, B):
        super().__init__(A=A, B=B)

    def todense(self):
        A, B = self.args
        return A.todense() @ B.todense()


class _ScaledLinearOperator(scipy.sparse.linalg.interface._ScaledLinearOperator, LinearOperator):
    """Linear operator scaled with a scalar."""

    def __init__(self, A, alpha):
        super().__init__(A=A, alpha=alpha)

    def todense(self):
        A, alpha = self.args
        return alpha * A.todense()


class _PowerLinearOperator(scipy.sparse.linalg.interface._PowerLinearOperator, LinearOperator):
    """Linear operator raised to a non-negative integer power."""

    def __init__(self, A, p):
        super().__init__(A=A, p=p)


class Identity(LinearOperator):
    """
    The identity operator.
    """

    def __init__(self, shape, dtype=None):
        # Check shape
        if np.isscalar(shape):
            _shape = (shape, shape)
        elif shape[0] != shape[1]:
            raise ValueError("The identity operator must be square.")
        else:
            _shape = shape
        # Initiator of super class
        super(Identity, self).__init__(dtype=dtype, shape=_shape)

    def _matvec(self, x):
        return x

    def _rmatvec(self, x):
        return x

    def _rmatmat(self, x):
        return x

    def _matmat(self, x):
        return x

    def _adjoint(self):
        return self

    def todense(self):
        return np.eye(N=self.shape[0], M=self.shape[1])

    def rank(self):
        return self.shape[0]

    def eigvals(self):
        return np.ones(self.shape[0])

    def cond(self, p=None):
        return 1.

    def det(self):
        return 1.

    def logabsdet(self):
        return 0.

    def trace(self):
        return self.shape[0]


class Diagonal(LinearOperator):
    """
    A linear operator representing the diagonal from another linear operator.

    Parameters
    ----------
    Op : LinearOperator
        Linear operator of which to represent the diagonal.
    """

    # TODO: should this be an operator itself or a function of a LinearOperator?
    #   - a function allows subclasses (e.g. MatrixMult) to implement more efficient versions than n products e_i A e_i
    def __init__(self, Op):
        raise NotImplementedError


class MatrixMult(scipy.sparse.linalg.interface.MatrixLinearOperator, LinearOperator):
    """
    A linear operator defined via a matrix.

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
        sign, logdet = np.linalg.slogdet(self.A)
        return logdet

    def trace(self):
        return np.trace(self.A)


class Vec(LinearOperator):
    """
    Vectorization operator.

    The column- or row-wise vectorization operator stacking the columns or rows of a matrix representation of a
    linear operator into a vector.

    Parameters
    ----------
    order : str
        Stacking order to apply. One of ``row`` or ``col``. Defaults to column-wise stacking.
    dim : int
        Either number of rows or columns, depending on the vectorization ``order``.
    """

    def __init__(self, order="col", dim=None):
        if order not in ["col", "row"]:
            raise ValueError("Not a valid stacking order. Choose one of 'col' or 'row'.")
        self.mode = order
        super().__init__(dtype=float, shape=(dim, dim))

    def _matvec(self, x):
        """Vectorization of a vector is the identity."""
        return x

    def _matmat(self, X):
        """Stacking of matrix rows or columns."""
        if self.mode == "row":
            return X.ravel(order='C')
        else:
            return X.ravel(order='F')


class Vec2Svec(LinearOperator):
    """
    Symmetric vectorization operator.

    The column- or row-wise symmetric normalized vectorization operator :math:`\\operatorname{svec}` [1]_ stacking the
    (normalized) lower/upper triangular components of a symmetric matrix of a linear operator into a
    vector. It is defined by

    .. math::
        \\operatorname{svec}(S) = \\begin{bmatrix}
                                    S_{11} \\\\
                                    \\sqrt{2} S_{21} \\\\
                                    \\vdots \\\\
                                    \\sqrt{2} S_{n1} \\\\
                                    S_{22} \\\\
                                    \\sqrt{2} S_{32} \\\\
                                    \\vdots \\\\
                                    \\sqrt{2} S_{n2} \\\\
                                    \\vdots \\\\
                                    S_{nn}
                                  \\end{bmatrix}

    where :math:`S` is a symmetric linear operator defined on :math:`\\mathbb{R}^n`.

    References
    ----------
    .. [1] De Klerk, E., Aspects of Semidefinite Programming, *Kluwer Academic Publishers*, 2002

    Notes
    -----
    It holds that :math:`Q\\operatorname{svec}(S) = \\operatorname{vec}(S)`, where :math:`Q` is a unique matrix with
    orthonormal rows.

    Parameters
    ----------
    dim : int
        Dimension of the symmetric matrix to be reshaped.
    check_symmetric : bool, default=False
        Check whether the given matrix or vector corresponds to a symmetric matrix argument. Note, this option can slow
        down performance.
    """

    def __init__(self, dim, check_symmetric=False):
        self._dim = dim
        self.check_symmetric = check_symmetric
        super().__init__(dtype=float, shape=(int(0.5 * dim * (dim + 1)), dim * dim))

    def _matvec(self, x):
        """Assumes x = vec(X)."""
        X = np.reshape(x.copy(), (self._dim, self._dim))
        if self.check_symmetric:
            assert (X.T == X).all(), "Given vector does not correspond to a symmetric matrix."

        X[np.triu_indices(self._dim, k=1)] *= np.sqrt(2)
        ind = np.triu_indices(self._dim, k=0)
        return X[ind]

    def _matmat(self, X):
        return self._matvec(X.ravel())


def _vec2svec(n):
    """
    Linear map from :math:`\\operatorname{vec}(S)` to :math:`\\operatorname{svec}(S)`

    Defines the unique matrix :math:`Q` with orthonormal rows such that
    :math:`\\operatorname{svec}(S) = Q\\operatorname{vec}(S)` [1]_ used to efficiently compute the symmetric Kronecker
    product.

    References
    ----------
    .. [1] De Klerk, E., Aspects of Semidefinite Programming, *Kluwer Academic Publishers*, 2002

    Parameters
    ----------
    n : int
        Dimension of the symmetric matrix :math:`S`.

    Returns
    -------
    Q : scipy.spmatrix
        Sparse array representing :math:`Q`.

    """
    # TODO: these pairwise comparisons are extremely inefficient, find a better implementation without building Q by
    #   directly implementing the vectorwise definition. By making this function into a linear operator we can still
    #   obtain a dense representation if necessary.
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Dimension of the input matrix S must be a positive integer.")

    # Get svec and vec indices
    cind, rind = np.triu_indices(n=n, k=0)
    rind_full, cind_full = np.indices(
        dimensions=(n, n))
    rind_full = rind_full.ravel()
    cind_full = cind_full.ravel()

    # Define entries with 1
    rows1 = np.nonzero(rind == cind)[0]
    cols1 = np.nonzero(rind_full == cind_full)[0]
    entries1 = np.ones_like(rows1)

    # Define entries with sqrt(2)/2
    # see also: Schaecke, K., On the Kronecker product. Master's thesis, University of Waterloo, 2004
    boolmask1 = np.equal.outer(rind, rind_full) & np.equal.outer(cind, cind_full) & np.not_equal.outer(cind, rind_full)
    boolmask2 = np.equal.outer(rind, cind_full) & np.equal.outer(cind, rind_full) & np.not_equal.outer(cind, cind_full)
    boolmask = boolmask1 | boolmask2
    rowsS2, colsS2 = np.nonzero(boolmask)
    entriesS2 = np.sqrt(2) / 2 * np.ones_like(rowsS2)

    # Build sparse matrix from row and column indices
    data = np.concatenate([entries1, entriesS2])
    row_ind = np.concatenate([rows1, rowsS2])
    col_ind = np.concatenate([cols1, colsS2])
    return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(int(0.5 * n * (n + 1)), n ** 2), dtype=float)


class Symmetrize(LinearOperator):
    """
    Symmetrizes a vector in its matrix representation.

    Given a vector x=vec(X) representing a square matrix X, this linear operator computes y=vec(1/2(X + X^T)).

    Parameters
    ----------
    dim : int
        Dimension of matrix X.
    """

    def __init__(self, dim):
        self._dim = dim
        super().__init__(dtype=float, shape=(dim * dim, dim * dim))

    def _matvec(self, x):
        """Assumes x=vec(X)."""
        X = np.reshape(x.copy(), (self._dim, self._dim))
        Y = 0.5 * (X + X.T)
        return Y.reshape(-1, 1)


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

    References
    ----------
    .. [1] Van Loan, C. F., The ubiquitous Kronecker product, *Journal of Computational and Applied Mathematics*, 2000,
            123, 85-100

    Parameters
    ----------
    A : np.ndarray or LinearOperator
        The first operator.
    B : np.ndarray or LinearOperator
        The second operator.
    dtype : dtype
        Data type of the operator.

    See Also
    --------
    SymmetricKronecker : The symmetric Kronecker product of two linear operators.

    """

    # todo: extend this to list of operators
    def __init__(self, A, B, dtype=None):
        self.A = aslinop(A)
        self.B = aslinop(B)
        super().__init__(dtype=dtype, shape=(self.A.shape[0] * self.B.shape[0],
                                             self.A.shape[1] * self.B.shape[1]))

    def _matvec(self, X):
        """
        Efficient multiplication via (A (x) B)vec(X) = vec(AXB^T) where vec is the row-wise vectorization operator.
        """
        X = X.reshape(self.A.shape[1], self.B.shape[1])
        Y = self.B.matmat(X.T)
        return self.A.matmat(Y.T).ravel()

    def _rmatvec(self, X):
        """
        Based on (A (x) B)^T = A^T (x) B^T.
        """
        X = X.reshape(self.A.shape[0], self.B.shape[0])
        Y = self.B.H.matmat(X.T)
        return self.A.H.matmat(Y.T).ravel()


class SymmetricKronecker(LinearOperator):
    """
    Symmetric Kronecker product of two linear operators.

    The symmetric Kronecker product [1]_ :math:`A \\otimes_{s} B` of two square linear operators :math:`A` and
    :math:`B` maps a symmetric linear operator :math:`X` to :math:`\\mathbb{R}^{\\frac{1}{2}n (n+1)}`. It is given by

    .. math::
        (A \\otimes_{s} B)\\operatorname{svec}(X) = \\frac{1}{2} \\operatorname{svec}(AXB^{\\top} + BXA^{\\top})

    where :math:`\\operatorname{svec}(X) = (X_{11}, \\sqrt{2} X_{12}, \\dots, X_{1n}, X_{22}, \\sqrt{2} X_{23},
    \\dots, \\sqrt{2}X_{2n}, \\dots X_{nn})^{\\top}` is the (row-wise, normalized) symmetric stacking operator. The
    implementation is based on the relationship :math:`Q^\\top \\operatorname{svec}(X) = \\operatorname{vec}(X)` with an
    orthonormal matrix :math:`Q` [2]_.

    References
    ----------
    .. [1] Van Loan, C. F., The ubiquitous Kronecker product, *Journal of Computational and Applied Mathematics*, 2000,
            123, 85-100
    .. [2] De Klerk, E., Aspects of Semidefinite Programming, *Kluwer Academic Publishers*, 2002

    Note
    ----
    The symmetric Kronecker product has a symmetric matrix representation if both :math:`A` and :math:`B` are symmetric.

    See Also
    --------
    Kronecker : The Kronecker product of two linear operators.
    """

    # TODO: update documentation to map from n2xn2 to matrices of rank 1/2n(n+1), representation symmetric n2xn2

    def __init__(self, A, B=None, dtype=None):
        # Set parameters
        self.A = aslinop(A)
        self._ABequal = False
        if B is None:
            self.B = self.A
            self._ABequal = True
        else:
            self.B = aslinop(B)
        self._n = self.A.shape[0]
        if self.A.shape != self.B.shape or self.A.shape[1] != self._n:
            raise ValueError("Linear operators A and B must be square and have the same dimensions.")

        # Initiator of superclass
        super().__init__(dtype=dtype, shape=(self._n ** 2, self._n ** 2))

    def _matvec(self, x):
        """
        Efficient multiplication via (A (x)_s B)vec(X) = 1/2 vec(BXA^T + AXB^T) where vec is the column-wise normalized
        symmetric stacking operator.
        """
        # vec(x)
        X = x.reshape(self._n, self._n)

        # (A (x)_s B)vec(X) = 1/2 vec(BXA^T + AXB^T)
        Y1 = (self.A @ (self.B @ X).T).T
        Y2 = (self.B @ (self.A @ X).T).T
        Y = 0.5 * (Y1 + Y2)
        return Y.ravel()

    def _rmatvec(self, x):
        """Based on (A (x)_s B)^T = A^T (x)_s B^T."""
        # vec(x)
        X = x.reshape(self._n, self._n)

        # (A^T (x)_s B^T)vec(X) = 1/2 vec(B^T XA + A^T XB)
        Y1 = (self.A.H @ (self.B.H @ X).T).T
        Y2 = (self.B.H @ (self.A.H @ X).T).T
        Y = 0.5 * (Y1 + Y2)
        return Y.ravel()

    # TODO: add efficient implementation of _matmat based on (Symmetric) Kronecker properties

    def todense(self):
        """Dense representation of the symmetric Kronecker product"""
        # 1/2 (A (x) B + B (x) A)
        A_dense = self.A.todense()
        B_dense = self.B.todense()
        return 0.5 * (np.kron(A_dense, B_dense) + np.kron(B_dense, A_dense))


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
    <2x3 MatrixMult with dtype=int32>
    """
    if isinstance(A, LinearOperator):
        return A
    # if isinstance(A, RandomVariable):
    #     # TODO: aslinearoperator also for random variables; change docstring example;
    #     #  not needed if LinearOperator inherits from RandomVariable
    #     # TODO: this causes a circular dependency between RandomVariable and LinearOperator
    #     raise NotImplementedError
    elif isinstance(A, (np.ndarray, scipy.sparse.spmatrix)):
        return MatrixMult(A=A)
    else:
        op = scipy.sparse.linalg.aslinearoperator(A)
        return LinearOperator(op, shape=op.shape)
