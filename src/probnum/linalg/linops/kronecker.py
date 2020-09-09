"""
Operators of Kronecker-type or related.

This module implements operators of Kronecker-type or linked to Kronecker-type products.
"""
import numpy as np

from probnum.linalg.linops.linearoperators import LinearOperator, aslinop


class Symmetrize(LinearOperator):
    """
    Symmetrizes a vector in its matrix representation.

    Given a vector x=vec(X) representing a square matrix X, this linear operator
    computes y=vec(1/2(X + X^T)).

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


class Vec(LinearOperator):
    """
    Vectorization operator.

    The column- or row-wise vectorization operator stacking the columns or rows of a
    matrix representation of a linear operator into a vector.

    Parameters
    ----------
    order : str
        Stacking order to apply. One of ``row`` or ``col``. Defaults to column-wise
        stacking.
    dim : int
        Either number of rows or columns, depending on the vectorization ``order``.
    """

    def __init__(self, order="col", dim=None):
        if order not in ["col", "row"]:
            raise ValueError(
                "Not a valid stacking order. Choose one of 'col' or 'row'."
            )
        self.mode = order
        super().__init__(dtype=float, shape=(dim, dim))

    def _matvec(self, x):
        """Vectorization of a vector is the identity."""
        return x

    def _matmat(self, X):
        """Stacking of matrix rows or columns."""
        if self.mode == "row":
            return X.ravel(order="C")
        else:
            return X.ravel(order="F")


class Svec(LinearOperator):
    """
    Symmetric vectorization operator.

    The column- or row-wise symmetric normalized vectorization operator
    :math:`\\operatorname{svec}` [1]_ stacking the (normalized) lower/upper triangular
    components of a symmetric matrix of a linear operator into a vector. It is defined
    by

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

    Parameters
    ----------
    dim : int
        Dimension of the symmetric matrix to be reshaped.
    check_symmetric : bool, default=False
        Check whether the given matrix or vector corresponds to a symmetric matrix
        argument. Note, this option can slow down performance.

    Notes
    -----
    It holds that :math:`Q\\operatorname{svec}(S) = \\operatorname{vec}(S)`, where
    :math:`Q` is a unique matrix with orthonormal rows.

    References
    ----------
    .. [1] De Klerk, E., Aspects of Semidefinite Programming, *Kluwer Academic
       Publishers*, 2002
    """

    def __init__(self, dim, check_symmetric=False):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                "Dimension of the input matrix S must be a positive integer."
            )
        self._dim = dim
        self.check_symmetric = check_symmetric
        super().__init__(dtype=float, shape=(dim * dim, int(0.5 * dim * (dim + 1))))

    def _matvec(self, x):
        """Assumes x = vec(X)."""
        X = np.reshape(x.copy(), (self._dim, self._dim))
        if self.check_symmetric and not (X.T == X).all():
            raise ValueError(
                "The given vector does not correspond to a symmetric matrix."
            )

        X[np.triu_indices(self._dim, k=1)] *= np.sqrt(2)
        ind = np.triu_indices(self._dim, k=0)
        return X[ind]

    def _matmat(self, X):
        """
        Vectorizes X if of dimension n^2, otherwise applies Svec to each column of X.
        """
        if np.shape(X)[0] == np.shape(X)[1] == self._dim:
            return self._matvec(X.ravel())
        elif np.shape(X)[0] == self._dim * self._dim:
            return np.hstack([self._matvec(col.reshape(-1, 1)) for col in X.T])
        else:
            raise ValueError(
                "Dimension mismatch. Argument must be either a (n x n) matrix or "
                "(n^2 x k)"
            )


class Kronecker(LinearOperator):
    """
    Kronecker product of two linear operators.

    The Kronecker product [1]_ :math:`A \\otimes B` of two linear operators :math:`A`
    and :math:`B` is given by

    .. math::
        A \\otimes B = \\begin{bmatrix}
            A_{11} B   &  \\dots   & A_{1 m_1} B  \\\\
            \\vdots     &  \\ddots  & \\vdots \\\\
            A_{n_11} B &  \\dots   & A_{n_1 m_1} B
        \\end{bmatrix}

    where :math:`A_{ij}v=A(v_j e_i)`, where :math:`e_i` is the :math:`i^{\\text{th}}`
    unit vector. The result is a new linear operator mapping from
    :math:`\\mathbb{R}^{n_1n_2}` to :math:`\\mathbb{R}^{m_1m_2}`. By recognizing that
    :math:`(A \\otimes B)\\operatorname{vec}(X) = AXB^{\\top}`, the Kronecker product
    can be understood as "translation" between matrix multiplication and (row-wise)
    vectorization.

    Parameters
    ----------
    A : np.ndarray or LinearOperator
        The first operator.
    B : np.ndarray or LinearOperator
        The second operator.
    dtype : dtype
        Data type of the operator.

    References
    ----------
    .. [1] Van Loan, C. F., The ubiquitous Kronecker product, *Journal of Computational
        and Applied Mathematics*, 2000, 123, 85-100

    See Also
    --------
    SymmetricKronecker : The symmetric Kronecker product of two linear operators.

    """

    # todo: extend this to list of operators
    def __init__(self, A, B, dtype=None):
        self.A = aslinop(A)
        self.B = aslinop(B)
        super().__init__(
            dtype=dtype,
            shape=(
                self.A.shape[0] * self.B.shape[0],
                self.A.shape[1] * self.B.shape[1],
            ),
        )

    def _matvec(self, X):
        """
        Efficient multiplication via (A (x) B)vec(X) = vec(AXB^T) where vec is the
        row-wise vectorization operator.
        """
        X = X.reshape(self.A.shape[1], self.B.shape[1])
        Y = self.B.matmat(X.T)
        return self.A.matmat(Y.T).ravel()

    def _rmatvec(self, X):
        # (A (x) B)^T = A^T (x) B^T.
        X = X.reshape(self.A.shape[0], self.B.shape[0])
        Y = self.B.H.matmat(X.T)
        return self.A.H.matmat(Y.T).ravel()

    def transpose(self):
        # (A (x) B)^T = A^T (x) B^T
        return Kronecker(A=self.A.transpose(), B=self.B.transpose(), dtype=self.dtype)

    def inv(self):
        #  (A (x) B)^-1 = A^-1 (x) B^-1
        return Kronecker(A=self.A.inv(), B=self.B.inv(), dtype=self.dtype)

    # Properties
    def rank(self):
        return self.A.rank() * self.B.rank()

    def eigvals(self):
        raise NotImplementedError

    def cond(self, p=None):
        return self.A.cond(p=p) * self.B.cond(p=p)

    def det(self):
        # If A (m x m) and B (n x n), then det(A (x) B) = det(A)^n * det(B) * m
        if self.A.shape[0] == self.A.shape[1] and self.B.shape[0] == self.B.shape[1]:
            return self.A.det() ** self.B.shape[0] * self.B.det() ** self.A.shape[0]
        else:
            raise NotImplementedError

    def logabsdet(self):
        # If A (m x m) and B (n x n), then det(A (x) B) = det(A)^n * det(B) * m
        if self.A.shape[0] == self.A.shape[1] and self.B.shape[0] == self.B.shape[1]:
            return (
                self.B.shape[0] * self.A.logabsdet()
                + self.A.shape[0] * self.B.logabsdet()
            )
        else:
            raise NotImplementedError

    def trace(self):
        if self.A.shape[0] == self.A.shape[1] and self.B.shape[0] == self.B.shape[1]:
            return self.A.trace() * self.B.trace()
        else:
            raise NotImplementedError


class SymmetricKronecker(LinearOperator):
    """
    Symmetric Kronecker product of two linear operators.

    The symmetric Kronecker product [1]_ :math:`A \\otimes_{s} B` of two square linear
    operators :math:`A` and :math:`B` maps a symmetric linear operator :math:`X` to
    :math:`\\mathbb{R}^{\\frac{1}{2}n (n+1)}`. It is given by

    .. math::
        (A \\otimes_{s} B)\\operatorname{svec}(X) = \\frac{1}{2} \\operatorname{svec}(AXB^{\\top} + BXA^{\\top})

    where :math:`\\operatorname{svec}(X) = (X_{11}, \\sqrt{2} X_{12}, \\dots, X_{1n},
    X_{22}, \\sqrt{2} X_{23}, \\dots, \\sqrt{2}X_{2n}, \\dots X_{nn})^{\\top}` is the
    (row-wise, normalized) symmetric stacking operator. The implementation is based on
    the relationship :math:`Q^\\top \\operatorname{svec}(X) = \\operatorname{vec}(X)`
    with an orthonormal matrix :math:`Q` [2]_.

    Note
    ----
    The symmetric Kronecker product has a symmetric matrix representation if both
    :math:`A` and :math:`B` are symmetric.

    References
    ----------
    .. [1] Van Loan, C. F., The ubiquitous Kronecker product, *Journal of Computational
       and Applied Mathematics*, 2000, 123, 85-100
    .. [2] De Klerk, E., Aspects of Semidefinite Programming, *Kluwer Academic
       Publishers*, 2002

    See Also
    --------
    Kronecker : The Kronecker product of two linear operators.
    """  # pylint: disable=line-too-long

    # TODO: update documentation to map from n2xn2 to matrices of rank 1/2n(n+1),
    # representation symmetric n2xn2

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
            raise ValueError(
                "Linear operators A and B must be square and have the same dimensions."
            )

        # Initiator of superclass
        super().__init__(dtype=dtype, shape=(self._n ** 2, self._n ** 2))

    def _matvec(self, x):
        """
        Efficient multiplication via (A (x)_s B)vec(X) = 1/2 vec(BXA^T + AXB^T) where
        vec is the column-wise normalized symmetric stacking operator.
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

    # TODO: add efficient implementation of _matmat based on (Symmetric) Kronecker
    # properties

    def todense(self):
        """Dense representation of the symmetric Kronecker product"""
        # 1/2 (A (x) B + B (x) A)
        A_dense = self.A.todense()
        B_dense = self.B.todense()
        return 0.5 * (np.kron(A_dense, B_dense) + np.kron(B_dense, A_dense))

    def inv(self):
        # (A (x)_s A)^-1 = A^-1 (x)_s A^-1
        if self._ABequal:
            return SymmetricKronecker(A=self.A.inv(), dtype=self.dtype)
        else:
            return NotImplementedError
