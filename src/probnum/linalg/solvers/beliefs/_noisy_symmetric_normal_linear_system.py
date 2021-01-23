"""Belief over a linear system with noise-corrupted system matrix."""

from typing import Optional, Union

import numpy as np

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers.hyperparams import LinearSystemNoise
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

from ._symmetric_normal_linear_system import SymmetricNormalLinearSystemBelief

# pylint: disable="invalid-name"


# Public classes and functions. Order is reflected in documentation.
__all__ = ["NoisySymmetricNormalLinearSystemBelief"]


class NoisySymmetricNormalLinearSystemBelief(SymmetricNormalLinearSystemBelief):
    r"""Belief over a noise-corrupted linear system.

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
    hyperparams :
        Noise on the system matrix and right hand side.
    """

    def __init__(
        self,
        x: rvs.Normal,
        A: rvs.Normal,
        Ainv: rvs.Normal,
        b: Union[rvs.Constant, rvs.Normal],
        hyperparams: LinearSystemNoise = LinearSystemNoise(A_eps=None, b_eps=None),
    ):
        super().__init__(x=x, A=A, Ainv=Ainv, b=b, hyperparams=hyperparams)

    @classmethod
    def from_solution(
        cls,
        x0: np.ndarray,
        problem: LinearSystem,
        check_for_better_x0: bool = True,
        hyperparams: Optional[LinearSystemNoise] = None,
    ) -> "NoisySymmetricNormalLinearSystemBelief":

        x0, Ainv0, A0, b0 = cls._belief_means_from_solution(
            x0=x0, problem=problem, check_for_better_x0=check_for_better_x0
        )
        Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))
        A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

        return cls(
            x=rvs.Normal(
                mean=x0, cov=cls._induced_solution_cov(Ainv=Ainv, b=problem.b)
            ),
            Ainv=Ainv,
            A=A,
            b=rvs.asrandvar(b0),
            hyperparams=hyperparams,
        )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
        hyperparams: Optional[LinearSystemNoise] = None,
    ) -> "NoisySymmetricNormalLinearSystemBelief":
        r"""Construct a belief about the linear system from an approximate inverse.

        Returns a belief about the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner. By default this internally
        inverts (the prior mean of) :math:`H_0`, which may be computationally costly.

        Parameters
        ----------
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        hyperparams :
            Noise on the system matrix and right hand side.
        """
        if not isinstance(Ainv0, rvs.Normal):
            Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))
        else:
            Ainv = Ainv0

        try:
            A0 = Ainv0.inv()
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) Ainv. "
                "Additionally, specify an inverse prior (mean) instead or wrap into "
                "a linear operator with an .inv() function."
            ) from exc
        A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv, b=problem.b),
            Ainv=Ainv,
            A=A,
            b=rvs.asrandvar(problem.b),
            hyperparams=hyperparams,
        )

    @classmethod
    def from_matrix(
        cls,
        A0: MatrixArgType,
        problem: LinearSystem,
        hyperparams: Optional[LinearSystemNoise] = None,
    ) -> "NoisySymmetricNormalLinearSystemBelief":
        r"""Construct a belief about the linear system from an approximate system matrix.

        Returns a belief about the linear system from an approximation of
        the system matrix :math:`A_0\approx A`. This internally inverts (the prior mean
        of) :math:`A_0`, which may be computationally costly.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        problem :
            Linear system to solve.
        hyperparams :
            Noise on the system matrix and right hand side.
        """
        if not isinstance(A0, rvs.Normal):
            A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
        else:
            A = A0

        try:
            Ainv0 = A0.inv()
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) A. "
                "Additionally, specify an inverse prior (mean) instead or wrap into "
                "a linear operator with an .inv() function."
            ) from exc
        Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv, b=problem.b),
            Ainv=Ainv,
            A=A,
            b=rvs.asrandvar(problem.b),
            hyperparams=hyperparams,
        )

    @classmethod
    def from_matrices(
        cls,
        A0: MatrixArgType,
        Ainv0: MatrixArgType,
        problem: LinearSystem,
        hyperparams: Optional[LinearSystemNoise] = None,
    ) -> "NoisySymmetricNormalLinearSystemBelief":
        r"""Construct a belief from an approximate system matrix and
        corresponding inverse.

        Returns a belief about the linear system from an approximation of
        the system matrix :math:`A_0\approx A` and an approximate inverse
        :math:`H_0\approx A^{-1}`.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        hyperparams :
            Noise on the system matrix and right hand side.
        """
        if not isinstance(A0, rvs.Normal):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
        if not isinstance(Ainv0, rvs.Normal):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=rvs.asrandvar(problem.b),
            hyperparams=hyperparams,
        )
