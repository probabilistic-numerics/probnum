"""Linear system beliefs.

Classes representing probabilistic (prior) beliefs over the quantities
of interest of a linear system such as its solution, the matrix inverse
or spectral information.
"""

import dataclasses
from typing import Union

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSystemBelief", "WeakMeanCorrespondence"]

# pylint: disable="invalid-name"


@dataclasses.dataclass
class LinearSystemBelief:
    r"""Belief over quantities of interest of a linear system.

    Random variables :math:`(\mathsf{x}, \mathsf{A}, \mathsf{H}, \mathsf{b})` modelling
    the solution :math:`x`, the system matrix :math:`A`, its (pseudo-)inverse
    :math:`H=A^{-1}` and the right hand side :math:`b` of a linear system :math:`Ax=b`.

    Parameters
    ----------
    x :
        Belief over the solution.
    A :
        Belief over the system matrix.
    Ainv :
        Belief over the (pseudo-)inverse of the system matrix.
    b :
        Belief over the right hand side

    Examples
    --------

    Construct a linear system belief from a preconditioner.

    >>>

    """
    x: rvs.RandomVariable
    A: rvs.RandomVariable
    Ainv: rvs.RandomVariable
    b: rvs.RandomVariable

    def __post_init__(self):
        # Check and normalize shapes
        if self.b.ndim == 1:
            self.b = self.b.reshape((-1, 1))
        if self.x.ndim == 1:
            self.x = self.x.reshape((-1, 1))
        if self.x.ndim != 2:
            raise ValueError("Belief over solution must be two-dimensional.")
        if self.A.ndim != 2 or self.Ainv.ndim != 2 or self.b.ndim != 2:
            raise ValueError("Beliefs over system components must be two-dimensional.")

        # Check shape mismatch
        def dim_mismatch_error(arg0, arg1, arg0_name, arg1_name):
            return ValueError(
                f"Dimension mismatch. The shapes of {arg0_name} : {arg0.shape} "
                f"and {arg1_name} : {arg1.shape} must match."
            )

        if self.A.shape[0] != self.b.shape[0]:
            raise dim_mismatch_error(self.A, self.b, "A", "b")

        if self.A.shape[0] != self.x.shape[0]:
            raise dim_mismatch_error(self.A, self.x, "A", "x")

        if self.x.shape[1] != self.b.shape[1]:
            raise dim_mismatch_error(self.x, self.b, "x", "b")

        if self.A.shape != self.Ainv.shape:
            raise dim_mismatch_error(self.A, self.Ainv, "A", "Ainv")

    # TODO: add different classmethods here to construct standard beliefs, i.e. from
    #  deterministic arguments (preconditioner), from a prior on the solution,
    #  from just an inverse prior, etc.

    @classmethod
    def from_solution(
        self,
        x0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> LinearSystem:
        """Construct a belief over the linear system from an approximate solution.

        Constructs a matrix-variate prior mean for :math:`H` from ``x0`` and ``b`` such
        that :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`.

        Parameters
        ----------
        x0 :
            Initial guess for the solution of the linear system.
        problem :
            Linear system to solve.

        Returns
        -------
        A0_mean :
            Mean of the matrix-variate prior distribution on the system matrix
            :math:`A`.
        Ainv0_mean :
            Mean of the matrix-variate prior distribution on the inverse of the system
            matrix :math:`H = A^{-1}`.
        """
        # Check inner product between x0 and b; if negative or zero, choose better
        # initialization

        bx0 = np.squeeze(problem.b.T @ x0)
        bb = np.linalg.norm(problem.b) ** 2
        if bx0 < 0:
            x0 = -x0
            bx0 = -bx0
            print("Better initialization found, setting x0 = - x0.")
        elif bx0 == 0:
            if np.all(problem.b == np.zeros_like(problem.b)):
                print("Right-hand-side is zero. Initializing with solution x0 = 0.")
                x0 = problem.b
            else:
                print("Better initialization found, setting x0 = (b'b/b'Ab) * b.")
                bAb = np.squeeze(problem.b.T @ (problem.A @ problem.b))
                x0 = bb / bAb * problem.b
                bx0 = bb ** 2 / bAb

        # Construct prior mean of A and H
        alpha = 0.5 * bx0 / bb

        def _mv(v):
            return (x0 - alpha * problem.b) * (x0 - alpha * problem.b).T @ v

        def _mm(M):
            return (x0 - alpha * problem.b) @ (x0 - alpha * problem.b).T @ M

        Ainv0_mean = linops.ScalarMult(
            scalar=alpha, shape=problem.A.shape
        ) + 2 / bx0 * linops.LinearOperator(
            matvec=_mv, matmat=_mm, shape=problem.A.shape
        )
        A0_mean = linops.ScalarMult(scalar=1 / alpha, shape=problem.A.shape) - 1 / (
            alpha * np.squeeze((x0 - alpha * problem.b).T @ x0)
        ) * linops.LinearOperator(matvec=_mv, matmat=_mm, shape=problem.A.shape)

        # TODO: what covariance should be returned for this prior mean?

    @classmethod
    def from_inverse(
        self, Ainv0: Union[np.ndarray, rvs.RandomVariable]
    ) -> LinearSystem:
        r"""Construct a belief over the linear system from an approximate inverse.

        Returns a belief over the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner.
        """
        raise NotImplementedError

    @classmethod
    def from_matrix(self, A0: Union[np.ndarray, rvs.RandomVariable]) -> LinearSystem:
        r"""Construct a belief over the linear system from an approximate system matrix.

        Returns a belief over the linear system from an approximation of
        the system matrix :math:`A_0\approx A`.
        """
        raise NotImplementedError


class WeakMeanCorrespondence(LinearSystemBelief):
    r"""Belief enforcing weak mean correspondence.

    Belief over the linear system such that the means over the matrix and its inverse
    correspond and the covariance symmetric Kronecker factors act like :math:`A` and the
    approximate inverse :math:`H_0` on the spaces spanned by the actions and
    observations. On the respective orthogonal spaces the uncertainty over the matrix
    and its inverse is determined by scaling parameters.

    For a scalar prior mean :math:`A_0 = H_0^{-1} = \alpha I`, when paired with a
    :class:`~probnum.linalg.linearsolvers.policies.ConjugateDirections`
    policy and linear observations, this (prior) belief recovers *the method of
    conjugate gradients*. [1]_

    For more details, see Wenger and Hennig, 2020. [1]_

    Parameters
    ----------
    x :
        Belief over the solution.
    A :
        Belief over the system matrix.
    Ainv :
        Belief over the (pseudo-)inverse of the system matrix.
    b :
        Belief over the right hand side
    unc_scale_A :
        Uncertainty scaling :math:`\Phi` of the belief over the matrix in the unexplored
        space :math:`\operatorname{span}(s_1, \dots, s_k)^\perp`.
    unc_scale_A :
        Uncertainty scaling :math:`\Psi` of the belief over the inverse in the
        unexplored space :math:`\operatorname{span}(y_1, \dots, y_k)^\perp`.

    Notes
    -----
    This construction fulfills *weak posterior correspondence* [1]_ meaning on the space
    spanned by the observations :math:`y_i` it holds that :math:`\mathbb{
    E}[A]^{-1}y = \mathbb{E}[H]y` for all :math:`y \in \operatorname{span}(y_1,
    \dots, y_k)`.

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020

    See Also
    --------
    LinearSystemBelief : Belief over quantities of interest of a linear system.

    Examples
    --------
    """

    def __init__(
        self, x, A, Ainv, b, unc_scale_A: float = 0.0, unc_scale_Ainv: float = 0.0
    ):
        self.unc_scale_A = unc_scale_A
        self.unc_scale_Ainv = unc_scale_Ainv
        super().__init__(x=x, A=A, Ainv=Ainv, b=b)

    def matvec_span_actions(self, s):
        """"""
        raise NotImplementedError

    def matvec_span_observations(self, y):
        """"""
        raise NotImplementedError
