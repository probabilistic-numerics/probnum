"""Gaussian linear system belief encoding symmetry of the system matrix."""

from typing import List, Optional, Union

import numpy as np

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

from ._linear_system import LinearSystemBelief

# Public classes and functions. Order is reflected in documentation.
__all__ = ["SymmetricNormalLinearSystemBelief"]

# pylint: disable="invalid-name"


class SymmetricNormalLinearSystemBelief(LinearSystemBelief):
    r"""Gaussian belief encoding symmetry of the system matrix and its inverse.

    Normally distributed random variables  :math:`(\mathsf{x}, \mathsf{A},
    \mathsf{H}, \mathsf{b})` modelling the solution :math:`x`, the system matrix
    :math:`A`, its (pseudo-)inverse :math:`H=A^{-1}` and the right hand side
    :math:`b` of a linear system :math:`Ax=b`. This belief encodes symmetry of

    .. math::
        \mathsf{A} &\sim \mathcal{N}(A_0, W_0^{\mathsf{A}} \otimes_s W_0^{\mathsf{A}})\\
        \mathsf{H} &\sim \mathcal{N}(H_0, W_0^{\mathsf{H}} \otimes_s W_0^{\mathsf{H}})

    via :class:`~probnum.linops.SymmetricKronecker` product structured covariances. [#]_

    Parameters
    ----------
    x :
        Belief over the solution.
    A :
        Belief over the system matrix.
    Ainv :
        Belief over the (pseudo-)inverse of the system matrix.
    b :
        Belief over the right hand side.

    References
    ----------
    .. [#] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
           Optimization*, 2015, 25, 234-260
    """

    def __init__(
        self,
        x: rvs.Normal,
        A: rvs.Normal,
        Ainv: rvs.Normal,
        b: Union[rvs.Constant, rvs.Normal],
    ):
        super().__init__(x=x, Ainv=Ainv, A=A, b=b)

    @classmethod
    def from_solution(
        cls,
        x0: np.ndarray,
        problem: LinearSystem,
        check_for_better_x0: bool = True,
    ) -> "SymmetricNormalLinearSystemBelief":
        x0, Ainv0, A0 = cls._belief_means_from_solution(
            x0=x0, problem=problem, check_for_better_x0=check_for_better_x0
        )

        # If b = 0, set x0 = 0
        if check_for_better_x0 and np.all(x0 == problem.b):
            A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
            Ainv = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

            return cls(
                x=rvs.Normal(
                    mean=x0,
                    cov=linops.ScalarMult(
                        scalar=np.finfo(float).eps, shape=problem.A.shape
                    ),
                ),
                Ainv=Ainv,
                A=A,
                b=rvs.asrandvar(problem.b),
            )
        else:
            Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))
            A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=problem.A))

            return cls(
                x=rvs.Normal(
                    mean=x0, cov=cls._induced_solution_cov(Ainv=Ainv, b=problem.b)
                ),
                Ainv=Ainv,
                A=A,
                b=rvs.asrandvar(problem.b),
            )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
    ) -> "SymmetricNormalLinearSystemBelief":
        if not isinstance(Ainv0, rvs.Normal):
            Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))
        else:
            Ainv = Ainv0
        A = rvs.Normal(mean=problem.A, cov=linops.SymmetricKronecker(A=problem.A))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv, b=problem.b),
            Ainv=Ainv,
            A=A,
            b=rvs.asrandvar(problem.b),
        )

    @classmethod
    def from_matrix(
        cls,
        A0: MatrixArgType,
        problem: LinearSystem,
    ) -> "SymmetricNormalLinearSystemBelief":
        if not isinstance(A0, rvs.Normal):
            A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=problem.A))
        else:
            A = A0

        Ainv0 = linops.Identity(shape=A0.shape)
        Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv, b=problem.b),
            Ainv=Ainv,
            A=A,
            b=rvs.asrandvar(problem.b),
        )

    @classmethod
    def from_matrices(
        cls,
        A0: MatrixArgType,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
    ) -> "SymmetricNormalLinearSystemBelief":
        if not isinstance(A0, rvs.Normal):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=problem.A))
        if not isinstance(Ainv0, rvs.Normal):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=rvs.asrandvar(problem.b),
        )

    @staticmethod
    def _induced_solution_belief(Ainv: rvs.Normal, b: rvs.RandomVariable) -> rvs.Normal:
        r"""Induced belief about the solution from a belief about the inverse.

        Approximates the induced random variable :math:`x=Hb` for :math:`H \sim
        \mathcal{N}(H_0, W \otimes_s W)`, such that :math:`x \sim \mathcal{N}(\mu,
        \Sigma)` with :math:`\mu=\mathbb{E}[H]\mathbb{E}[b]` and :math:`\Sigma=\frac{
        1}{2}(Wb^\top Wb + Wb b^\top W)`.

        Parameters
        ----------
        Ainv :
            Belief over the (pseudo-)inverse of the system matrix.
        b :
            Belief over the right hand side
        """
        b = rvs.asrandvar(b)
        return rvs.Normal(
            mean=Ainv.mean @ b.mean,
            cov=SymmetricNormalLinearSystemBelief._induced_solution_cov(Ainv=Ainv, b=b),
        )

    @staticmethod
    def _induced_solution_cov(
        Ainv: rvs.Normal, b: rvs.RandomVariable
    ) -> linops.LinearOperator:
        r"""Induced covariance of the belief about the solution.

        Approximates the covariance :math:`\Sigma` of the induced random variable
        :math:`x=Hb` for :math:`H \sim \mathcal{N}(H_0, W \otimes_s W)` such that
        :math:`\Sigma=\frac{1}{2}(Wb^\top Wb + Wb b^\top W)`.

        Parameters
        ----------
        Ainv :
            Belief over the (pseudo-)inverse of the system matrix.
        b :
            Belief over the right hand side
        """
        # TODO extend this to the case of multiple right hand sides, where the
        #  covariance is given by Prop S4 of Wenger and Hennig, 2020:
        #  \Sigma = 1/2 (W \otimes BWB + WB \boxtimes B'W)
        b = rvs.asrandvar(b)
        Wb = Ainv.cov.A @ b.mean
        bWb = Wb.T @ b.mean

        def _mv(v):
            return 0.5 * (bWb.item() * Ainv.cov.A @ v + Wb @ (Wb.T @ v))

        x_cov = linops.LinearOperator(
            shape=Ainv.shape, dtype=float, matvec=_mv, matmat=_mv
        )
        # Efficient trace computation
        x_cov.trace = lambda: 0.5 * (
            Ainv.cov.A.trace() * np.trace(bWb) + np.trace(Wb.T @ Wb)
        )

        return x_cov
