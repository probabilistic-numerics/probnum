"""Kronecker-type operators.

This module implements operators of Kronecker type or linked to
Kronecker-type products.
"""
import numpy as np

from . import _linear_operator, _utils


class Symmetrize(_linear_operator.LinearOperator):
    """Symmetrizes a vector in its matrix representation.

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


class Kronecker(_linear_operator.LinearOperator):
    r"""Kronecker product of two linear operators.

    The Kronecker product [1]_ :math:`A \otimes B : \mathbb{R}^{n_1n_2} \to
    \mathbb{R}^{m_1m_2}` of two linear operators :math:`A`
    and :math:`B` is given by

    .. math::
        (A \otimes B)_{(ij)(kl)} = \begin{bmatrix}
            A_{11} B   &  \dots   & A_{1 n_1} B  \\
            \vdots     &  \ddots  & \vdots \\
            A_{m_11} B &  \dots   & A_{m_1 n_1} B
        \end{bmatrix}_{(ij)(kl)} = A_{ij}B_{kl}

    By recognizing that
    :math:`(A \otimes B)\operatorname{vec}(X) = BXA^{\top}`, the Kronecker product
    can be understood as translation between matrix multiplication and (column-wise)
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
        self.A = _utils.aslinop(A)
        self.B = _utils.aslinop(B)
        super().__init__(
            dtype=dtype,
            shape=(
                self.A.shape[0] * self.B.shape[0],
                self.A.shape[1] * self.B.shape[1],
            ),
        )

    def _matvec(self, X):
        """Efficient multiplication via (A (x) B)vec(X) = vec(AXB^T) where vec is the
        **row-wise** vectorization operator.
        """
        X = X.reshape(self.A.shape[1], self.B.shape[1])
        Y = self.B.matmat(X.T)
        return self.A.matmat(Y.T).ravel()

    def _rmatvec(self, X):
        # (A (x) B)^T = A^T (x) B^T.
        X = X.reshape(self.A.shape[0], self.B.shape[0])
        Y = self.B.H.matmat(X.T)
        return self.A.H.matmat(Y.T).ravel()

    def _transpose(self):
        # (A (x) B)^T = A^T (x) B^T
        return Kronecker(A=self.A.transpose(), B=self.B.transpose(), dtype=self.dtype)

    def adjoint(self):
        return Kronecker(A=self.A.adjoint(), B=self.B.adjoint(), dtype=self.dtype)

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


class BoxProduct(_linear_operator.LinearOperator):
    r"""Box product of two linear operators.

    The box product [1]_ :math:`A \boxtimes B : \mathbb{R}^{n_1n_2} \to \mathbb{R}^{
    m_1m_2}` of two linear operators :math:`A \in \mathbb{R}^{m_1 \times n_1}` and
    :math:`B \in \mathbb{R}^{m_2 \times n_2}` is given by

    .. math::
        (A \boxtimes B)_{(ij)(kl)} = A_{il}B_{jk}

    Alternatively it can be characterized via the property :math:`(A \boxtimes B)
    \operatorname{vec}(X) = \operatorname{vec}(B X^\top A^\top)`for :math:`X \in
    \mathbb{R}^{n_1 \times n_2}`.

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
    .. [1] Olsen, P.A. et al. Efficient automatic differentiation of matrix functions.
        *Recent Advances in Algorithmic Differentiation*, 2012, 71-81

    See Also
    --------
    Kronecker : The Kronecker product of two linear operators.
    """

    def __init__(self, A, B, dtype=None):
        self.A = _utils.aslinop(A)
        self.B = _utils.aslinop(B)
        super().__init__(
            dtype=dtype,
            shape=(
                self.A.shape[0] * self.B.shape[0],
                self.A.shape[1] * self.B.shape[1],
            ),
        )

    def _matvec(self, X):
        raise NotImplementedError


class SymmetricKronecker(_linear_operator.LinearOperator):
    """Symmetric Kronecker product of two linear operators.

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
    BoxProduct : The box product of two linear operators.
    """

    # pylint: disable=line-too-long

    # TODO: update documentation to map from n2xn2 to matrices of rank 1/2n(n+1),
    # representation symmetric n2xn2

    def __init__(self, A, B=None, dtype=None):
        # Set parameters
        self.A = _utils.aslinop(A)
        self._ABequal = False
        if B is None:
            self.B = self.A
            self._ABequal = True
        else:
            self.B = _utils.aslinop(B)
        self._n = self.A.shape[0]
        if self.A.shape != self.B.shape or self.A.shape[1] != self._n:
            raise ValueError(
                "Linear operators A and B must be square and have the same dimensions."
            )

        # Initiator of superclass
        super().__init__(dtype=dtype, shape=(self._n ** 2, self._n ** 2))

    def _matvec(self, x):
        """Efficient multiplication via (A (x)_s B)vec(X) = 1/2 vec(BXA^T + AXB^T) where
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
    #   properties.

    def todense(self):
        """Dense representation of the symmetric Kronecker product."""
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
