"""Finite dimensional linear operators.

This module defines classes and methods that implement finite dimensional linear operators. It can be used to do linear
algebra with (structured) matrices without explicitly representing them in memory. This often allows for the definition
of a more efficient matrix-vector product. Linear operators can be applied, added, multiplied, transposed, and more as
one would expect from matrix algebra.

Several algorithms in the :mod:`probnum.linalg` library are able to operate on :class:`LinearOperator` instances.
"""

import numpy as np
import scipy.sparse.linalg
from probnum.probability import RandomVariable
from scipy.sparse.linalg.interface import MatrixLinearOperator

__all__ = ["LinearOperator", "MatrixMult", "Identity", "Kronecker", "SymmetricKronecker", "aslinop"]


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

    This class wraps :class:`scipy.sparse.linalg.LinearOperator` to provide support for
    :class:`~probnum.RandomVariable` arguments to linear operators.

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
    >>> from probnum.linalg import LinearOperator
    >>> def mv(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator((2,2), matvec=mv)
    >>> A
    <2x2 _CustomLinearOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([ 2.,  3.])
    >>> A * np.ones(2)
    array([ 2.,  3.])

    """

    # Overload arithmetic operators to give access to implemented functions (e.g. todense())
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
        elif isinstance(x, RandomVariable):
            pass
        else:
            y = np.asarray(y)

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

    # TODO: implement operations (eigs, cond, det, logabsdet, trace, ...)


# TODO: inheritance from _TransposedLinearOperator causes dependency on scipy>=1.4, implement our own instead?
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
    """Product of two linear operators."""

    def __init__(self, A, B):
        super().__init__(A=A, B=B)

    def todense(self):
        A, B = self.args
        return A.todense() * B.todense()


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


class MatrixMult(MatrixLinearOperator, LinearOperator):
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


class Svec(LinearOperator):
    """
    Symmetric vectorization operator.

    The column- or row-wise symmetric normalized vectorization operator :math:`\\operatorname{svec}` [1]_ stacking the
    (normalized) lower/upper triangular components of a symmetric matrix representation of a linear operator into a
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

    .. [1] De Klerk, E., Aspects of Semidefinite Programming, *Kluwer Academic Publishers*, 2002

    Notes
    -----
    It holds that :math:`Q\\operatorname{svec}(S) = \\operatorname{vec}(S)`, where :math:`Q` is a unique matrix with
    orthonormal rows.
    """

    def __init__(self, dim):
        raise NotImplementedError


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
        self.A = aslinop(A)
        self.B = aslinop(B)
        super().__init__(dtype=dtype, shape=(self.A.shape[0] * self.B.shape[0],
                                             self.A.shape[1] * self.B.shape[1]))

    def _matvec(self, x):
        """
        Efficient multiplication via (A (x) B)vec(X) = vec(AXB^T) where vec is the row-wise vectorization operator.
        """
        x = x.reshape(self.A.shape[1], self.B.shape[1])
        y = self.B.matmat(x.T)
        return self.A.matmat(y.T).ravel()

    def _rmatvec(self, x):
        """
        Based on (A (x) B)^T = A^T (x) B^T.
        """
        x = x.reshape(self.A.shape[0], self.B.shape[0])
        y = self.B.H.matmat(x.T)
        return self.A.H.matmat(y.T).ravel()


def _vec2svec(n):
    """
    Linear map from :math:`\\operatorname{vec}(S)` to :math:`\\operatorname{svec}(S)`

    Defines the unique matrix :math:`Q` with orthonormal rows such that
    :math:`\\operatorname{svec}(S) = Q\\operatorname{vec}(S)` [1]_ used to efficiently compute the symmetric Kronecker
    product.

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

    # TODO: Make sure definition matches the one in "Hennig, P. and Osborne M. A., Probabilistic Numerics, 2020"
    def __init__(self, A, B, dtype=None):
        # Set parameters
        self.A = aslinop(A)
        self.B = aslinop(B)
        self._n = self.A.shape[0]
        if self.A.shape != self.B.shape or self.A.shape[1] != self._n:
            raise ValueError("Linear operators A and B must be square and have the same dimensions.")

        # Orthonormal Q: svec to vec mapping
        self._Q = _vec2svec(n=self._n)

        # Initiator of superclass
        n_symkron = int(0.5 * self._n * (self._n + 1))
        super().__init__(dtype=dtype, shape=(n_symkron, n_symkron))

    def _matvec(self, x):
        """
        Efficient multiplication via (A (x)_s B)svec(X) = 1/2 svec(BXA^T + AXB^T) where svec is the column-wise normalized
        symmetric stacking operator.
        """
        # Q^T svec(x) = vec(x)
        X = (self._Q.transpose() @ x).reshape(self._n, self._n)

        # (A (x)_s B)svec(X) = 1/2 Q vec(BXA^T + AXB^T)
        Y = self.B @ (self.A @ X).transpose()
        Y = Y + Y.transpose()
        return 0.5 * self._Q @ Y.ravel()

    def _rmatvec(self, x):
        """Based on (A (x)_s B)^T = A^T (x)_s B^T."""
        # Q^T svec(x) = vec(x)
        X = (self._Q.transpose() @ x).reshape(self._n, self._n)

        # (A^T (x)_s B^T)svec(X) = 1/2 Q vec(B^T XA + A^T XB)
        Y = self.B.transpose() @ (self.A.transpose() @ X).transpose()
        Y = Y + Y.transpose()
        return 0.5 * self._Q @ Y.ravel()

    # TODO: add efficient implementation of _matmat based on (Symmetric) Kronecker properties


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
        return MatrixMult(A=A)
    else:
        return LinearOperator(scipy.sparse.linalg.aslinearoperator(A))
