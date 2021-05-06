"""Belief over a linear system with noise-corrupted system matrix."""

from typing import Optional, Union

import numpy as np

from probnum import linops, randvars
from probnum.linalg.solvers.beliefs._symmetric_normal_linear_system import (
    SymmetricNormalLinearSystemBelief,
)
from probnum.linalg.solvers.hyperparams import LinearSystemNoise
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

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

    Examples
    --------

    """

    def __init__(
        self,
        A: randvars.Normal,
        Ainv: randvars.Normal,
        b: Union[randvars.Constant, randvars.Normal],
        x: Optional[randvars.Normal] = None,
        hyperparams: LinearSystemNoise = None,
    ):
        if hyperparams is None:
            eps = 10 ** -2
            n = A.shape[0]
            hyperparams = LinearSystemNoise(
                epsA_cov=linops.SymmetricKronecker(A=eps * A.cov.A),
                epsb_cov=linops.Scaling(factors=eps, shape=(n, n)),
            )

        super().__init__(x=x, A=A, Ainv=Ainv, b=b, hyperparams=hyperparams)

    @property
    def hyperparams(self) -> Optional[LinearSystemNoise]:
        """Additive Gaussian noise on the system matrix and / or right hand side."""
        return super().hyperparams

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
        Ainv = randvars.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))
        A = randvars.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

        return cls(
            x=x0,
            Ainv=Ainv,
            A=A,
            b=randvars.asrandvar(b0),
            hyperparams=hyperparams,
        )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: Union[MatrixArgType, randvars.RandomVariable],
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
        if not isinstance(Ainv0, randvars.Normal):
            Ainv = randvars.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))
        else:
            Ainv = Ainv0
            Ainv0 = Ainv.mean

        try:
            A0 = Ainv0.inv()
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) Ainv. "
                "Additionally, specify an inverse prior (mean) instead or wrap into "
                "a linear operator with an .inv() function."
            ) from exc
        A = randvars.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))

        return cls(
            x=None,
            Ainv=Ainv,
            A=A,
            b=randvars.asrandvar(problem.b),
            hyperparams=hyperparams,
        )

    @classmethod
    def from_matrix(
        cls,
        A0: Union[MatrixArgType, randvars.RandomVariable],
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
        if not isinstance(A0, randvars.Normal):
            A = randvars.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
        else:
            A = A0
            A0 = A.mean

        try:
            Ainv0 = A0.inv()
        except AttributeError as exc:
            raise TypeError(
                "Cannot efficiently invert (prior mean of) A. "
                "Additionally, specify an inverse prior (mean) instead or wrap into "
                "a linear operator with an .inv() function."
            ) from exc
        Ainv = randvars.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=None,
            Ainv=Ainv,
            A=A,
            b=randvars.asrandvar(problem.b),
            hyperparams=hyperparams,
        )

    @classmethod
    def from_matrices(
        cls,
        A0: Union[MatrixArgType, randvars.RandomVariable],
        Ainv0: Union[MatrixArgType, randvars.RandomVariable],
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
        if not isinstance(A0, randvars.Normal):
            A0 = randvars.Normal(mean=A0, cov=linops.SymmetricKronecker(A=A0))
        if not isinstance(Ainv0, randvars.Normal):
            Ainv0 = randvars.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=None,
            Ainv=Ainv0,
            A=A0,
            b=randvars.asrandvar(problem.b),
            hyperparams=hyperparams,
        )

    def _induced_solution_cov(self) -> Union[np.ndarray, linops.LinearOperator]:

        if isinstance(self.Ainv.cov, linops.SymmetricKronecker):
            return super()._induced_solution_cov()
        else:
            Vb = self.Ainv.cov.A.A @ self.b.mean
            bVb = (Vb.T @ self.b.mean).item()

            Wb = self.Ainv.cov.B.A @ self.b.mean
            bWb = (Wb.T @ self.b.mean).item()

            def _matmul(x):
                return 0.5 * (
                    bVb * self.Ainv.cov.A.A @ x
                    + bWb * self.Ainv.cov.B.A @ x
                    + Vb @ (Vb.T @ x)
                    + Wb @ (Wb.T @ x)
                )

            x_cov = linops.LinearOperator(
                shape=self.Ainv.shape, dtype=float, matmul=_matmul
            )
            # Efficient trace computation
            x_cov.trace = lambda: 0.5 * (
                self.Ainv.cov.A.A.trace() * bVb
                + self.Ainv.cov.B.A.trace() * bWb
                + (Vb.T @ Vb).item()
                + (Wb.T @ Wb).item()
            )

            # TODO add correct covariance term from b here
            return x_cov
            # raise NotImplementedError
