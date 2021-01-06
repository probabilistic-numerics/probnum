"""Linear operators defining projections.

Projections :math:`P:V \rightarrow V` are linear transformations from a vector space
:math:`V` onto itself such that :math:`P^2=P`, i.e. applying the transformation twice is
equivalent to applying it once.
"""
from typing import Optional, Union

import numpy as np

from . import _linear_operator

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


class OrthogonalProjection(_linear_operator.LinearOperator):
    r"""Orthogonal Projection onto a subspace.

    Linear operator :math:`P:V \rightarrow V` projecting onto a subspace :math:`U
    \subset V` such that :math:`P^2 = P`. For an inner product on :math:`V` given by a
    symmetric positive definite matrix :math:`D`, the projection can be characterized as

    .. math::
      P_Ax = \operatorname{argmin}_{y \in \operatorname{range}(A)}\lVert x - y
      \rVert_D^2 = A(A^\top D A)^{-1}A^\top Dx

    where :math:`A` is the matrix whose columns are the basis of :math:`U`.

    See https://en.wikipedia.org/wiki/Projection_(linear_algebra) for more
    information.

    Parameters
    ----------
    subspace_basis :
        *shape=(n, k)* -- Matrix whose columns define the basis of the subspace to
        project on.
    is_orthonormal :
        Is the basis of the subspace orthonormal?
    innerprod_matrix :
        *shape=(n, n)* -- Symmetric positive definite matrix :math:`D` defining an
        inner product on :math:`V`.
    """

    def __init__(
        self,
        subspace_basis: np.ndarray,
        is_orthonormal=False,
        innerprod_matrix: Optional[
            Union[_linear_operator.LinearOperator, np.ndarray]
        ] = None,
    ):
        self.subspace_basis = subspace_basis
        self.is_orthonormal = is_orthonormal
        _shape = (subspace_basis.shape[0], subspace_basis.shape[0])
        if innerprod_matrix is None:
            innerprod_matrix = _linear_operator.Identity(shape=_shape)
        if not (
            innerprod_matrix.shape[0]
            == innerprod_matrix.shape[1]
            == subspace_basis.shape[0]
        ):
            raise ValueError(
                f"Shape of subspace basis {subspace_basis.shape} and "
                f"inner product defining matrix {innerprod_matrix.shape} "
                f"do not match."
            )
        self.innerprod_matrix = innerprod_matrix
        super().__init__(shape=_shape, dtype=float)

    @cached_property
    def _transformed_basis(self):
        r"""Transformed basis of the subspace.

        Compute the transformed basis :math:`A(A^\top D A)^{-1}` with
        respect to the inner product defined by the positive definite matrix
        :math:`D`."""
        if self.is_orthonormal:
            return self.subspace_basis
        else:
            return np.linalg.solve(
                self.subspace_basis.T @ (self.innerprod_matrix @ self.subspace_basis),
                self.subspace_basis.T,
            ).T

    def _matvec(self, x):
        return (
            self.subspace_basis
            @ self._transformed_basis.T
            @ (self.innerprod_matrix @ x)
        )

    def _matmat(self, X):
        return (
            self.subspace_basis
            @ self._transformed_basis.T
            @ (self.innerprod_matrix @ X)
        )

    def _transpose(self):
        if isinstance(self.innerprod_matrix, _linear_operator.Identity):
            return self
        else:

            def _matvec(x):
                return (
                    self.innerprod_matrix
                    @ self.subspace_basis
                    @ (self._transformed_basis.T @ x)
                )

            return _linear_operator.LinearOperator(
                matvec=_matvec,
                matmat=_matvec,
                shape=self.shape,
                dtype=float,
            )

    def inv(self):
        raise NotImplementedError

    # Properties
    def rank(self):
        return self.subspace_basis.shape[1]

    def eigvals(self):
        return np.array(self.rank() * [1.0] + (self.shape[0] - self.rank()) * [0.0])

    def cond(self, p=None):
        return np.inf if self.rank() != self.shape else 1.0

    def det(self):
        return 0.0 if self.rank() != self.shape else 1.0

    def logabsdet(self):
        return -np.inf if self.rank() != self.shape else 0.0

    def trace(self):
        if isinstance(self.innerprod_matrix, _linear_operator.Identity):
            return self.rank()
        else:
            return super().trace()
