"""Linear system belief.

Base class defining a belief over the quantities of interest of a linear
system such as its solution, the matrix inverse.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

try:
    # functools.cached_property is only available in Python >=3.8
    from functools import cached_property
except ImportError:
    from cached_property import cached_property

import probnum
import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.belief_updates import (
    SymMatrixNormalLinearObsBeliefUpdate,
)
from probnum.linalg.linearsolvers.observation_ops import MatrixMultObservation
from probnum.problems import LinearSystem

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSystemBelief",
]

# pylint: disable="invalid-name"


class LinearSystemBelief:
    """Belief over quantities of interest of a linear system.

    Random variables :math:`(\\mathsf{x}, \\mathsf{A}, \\mathsf{H}, \\mathsf{b})`
    modelling the solution :math:`x`, the system matrix :math:`A`, its (pseudo-)inverse
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
        Belief over the right hand side.

    Examples
    --------
    Construct a linear system belief from a preconditioner.

    >>> import numpy as np
    >>> from probnum.problems import LinearSystem
    >>> from probnum.linalg.linearsolvers.beliefs import LinearSystemBelief
    >>> # Linear System Ax=b
    >>> linsys = LinearSystem(
    ...     A=np.array([[7.5, 2.0, 1.0], [2.0, 2.0, 0.5], [1.0, 0.5, 5.5]]),
    ...     b=np.array([1.0, 2.0, -3.0]),
    ... )
    >>> # Preconditioner, i.e. approximate inverse of A
    >>> Ainv_approx = np.array(
    ...     [[ 0.2,   -0.18, -0.015],
    ...      [-0.18,   0.7,  -0.03],
    ...      [-0.015, -0.03,  0.20]]
    ... )
    >>> # Prior belief induced by approximate inverse
    >>> prior = LinearSystemBelief.from_inverse(Ainv0=Ainv_approx, problem=linsys)
    >>> # Initial residual Ax0 - b
    >>> residual = linsys.A @ prior.x.mean - linsys.b
    >>> residual
    array([[ 0.0825],
           [ 0.0525],
           [-0.1725]])
    """

    def __init__(
        self,
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
        hyperparams: Optional[Tuple[np.ndarray, ...]] = None,
    ):

        x, A, Ainv, b = self._reshape_2d(x=x, A=A, Ainv=Ainv, b=b)
        self._check_shape_mismatch(x=x, A=A, Ainv=Ainv, b=b)
        self._x = rvs.asrandvar(x)
        self._A = rvs.asrandvar(A)
        self._Ainv = rvs.asrandvar(Ainv)
        self._b = rvs.asrandvar(b)
        self._hyperparams = hyperparams

    @staticmethod
    def _reshape_2d(
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
    ) -> Tuple[
        rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable
    ]:
        """Reshape inputs to 2d matrices."""
        if b.ndim == 1:
            b = b.reshape((-1, 1))
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if x.ndim != 2:
            raise ValueError("Belief over solution must be two-dimensional.")
        if A.ndim != 2 or Ainv.ndim != 2 or b.ndim != 2:
            raise ValueError("Beliefs over system components must be two-dimensional.")
        return x, A, Ainv, b

    @staticmethod
    def _check_shape_mismatch(
        x: rvs.RandomVariable,
        A: rvs.RandomVariable,
        Ainv: rvs.RandomVariable,
        b: rvs.RandomVariable,
    ) -> None:
        """Check whether shapes of arguments match."""

        def dim_mismatch_error(arg0, arg1, arg0_name, arg1_name):
            return ValueError(
                f"Dimension mismatch. The shapes of {arg0_name} : {arg0.shape} "
                f"and {arg1_name} : {arg1.shape} must match."
            )

        if A.shape[0] != b.shape[0]:
            raise dim_mismatch_error(A, b, "A", "b")

        if A.shape[1] != x.shape[0]:
            raise dim_mismatch_error(A, x, "A", "x")

        if x.shape[1] != b.shape[1]:
            raise dim_mismatch_error(x, b, "x", "b")

        if A.shape != Ainv.shape:
            raise dim_mismatch_error(A, Ainv, "A", "Ainv")

    @cached_property
    def x(self) -> rvs.RandomVariable:
        """Belief over the solution."""
        if self._x is None and self.Ainv is not None and self.b is not None:
            return self._induced_solution_belief(Ainv=self.Ainv, b=self.b)
        else:
            return self._x

    @property
    def A(self) -> rvs.RandomVariable:
        """Belief over the system matrix."""
        return self._A

    @property
    def Ainv(self) -> rvs.RandomVariable:
        """Belief over the (pseudo-)inverse of the system matrix."""
        return self._Ainv

    @property
    def b(self) -> rvs.RandomVariable:
        """Belief over the right hand side."""
        return self._b

    @property
    def hyperparams(self) -> Optional[Tuple[np.ndarray, ...]]:
        """Hyperparameters of the belief."""
        return self._hyperparams

    @hyperparams.setter
    def hyperparams(self, values: Tuple[np.ndarray, ...]):
        """Set the hyperparameters of the belief."""
        self._hyperparams = values

    @classmethod
    def from_solution(
        cls,
        x0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
        check_for_better_x0: bool = True,
    ) -> "LinearSystemBelief":
        r"""Construct a belief over the linear system from an approximate solution.

        Constructs a matrix-variate prior mean for :math:`H` from an initial
        guess of the solution :math:`x0` and the right hand side :math:`b` such
        that :math:`H_0b = x_0`, :math:`H_0` symmetric positive definite and
        :math:`A_0 = H_0^{-1}`. If :code:`check_for_better_x0=True` and
        :math:`x_0^\top b \leq 0` the belief is initialized with a better approximate
        solution :math:`x_1` with lower error :math:`\lVert x_1 \rVert_A < \lVert x_0
        \rVert_A`.

        For a detailed construction see Proposition S5 of Wenger and Hennig,
        2020. [#]_

        Parameters
        ----------
        x0 :
            Initial guess for the solution of the linear system.
        problem :
            Linear system to solve.
        check_for_better_x0 :
            Choose a better initial guess for the solution if possible.

        References
        ----------
        .. [#] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for
               Machine Learning, *Advances in Neural Information Processing Systems (
               NeurIPS)*, 2020
        """
        if x0.ndim < 2:
            x0 = x0.reshape((-1, 1))

        # If b = 0, set x0 = 0
        if check_for_better_x0 and np.all(problem.b == np.zeros_like(problem.b)):
            x0 = problem.b
            A0 = linops.Identity(shape=problem.A.shape)
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
                b=rvs.Constant(support=problem.b),
            )
        else:
            bx0 = (problem.b.T @ x0).item()
            bb = np.linalg.norm(problem.b) ** 2
            # If inner product <x0, b> is non-positive, choose better initialization.
            if check_for_better_x0 and bx0 < -100 * np.finfo(float).eps:
                # <x0, b> < 0
                x0 = -x0
                bx0 = -bx0
            elif check_for_better_x0 and np.abs(bx0) < 100 * np.finfo(float).eps:
                # <x0, b> = 0, b != 0
                bAb = np.squeeze(problem.b.T @ (problem.A @ problem.b))
                x0 = bb / bAb * problem.b
                bx0 = bb ** 2 / bAb

            # Construct prior mean of A and Ainv
            alpha = 0.5 * bx0 / bb

            def _mv(v):
                return (x0 - alpha * problem.b) * (x0 - alpha * problem.b).T @ v

            def _mm(M):
                return (x0 - alpha * problem.b) @ (x0 - alpha * problem.b).T @ M

            Ainv_mean = linops.ScalarMult(
                scalar=alpha, shape=problem.A.shape
            ) + 2 / bx0 * linops.LinearOperator(
                matvec=_mv, matmat=_mm, shape=problem.A.shape
            )
            Ainv = rvs.Normal(
                mean=Ainv_mean, cov=linops.SymmetricKronecker(A=Ainv_mean)
            )

            A_mean = linops.ScalarMult(scalar=1 / alpha, shape=problem.A.shape) - 1 / (
                alpha * np.squeeze((x0 - alpha * problem.b).T @ x0)
            ) * linops.LinearOperator(matvec=_mv, matmat=_mm, shape=problem.A.shape)
            A = rvs.Normal(mean=A_mean, cov=linops.SymmetricKronecker(A=problem.A))

            return LinearSystemBelief(
                x=rvs.Normal(
                    mean=x0, cov=cls._induced_solution_cov(Ainv=Ainv, b=problem.b)
                ),
                Ainv=Ainv,
                A=A,
                b=rvs.Constant(support=problem.b),
            )

    @classmethod
    def from_inverse(
        cls,
        Ainv0: Union[np.ndarray, rvs.RandomVariable, linops.LinearOperator],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief over the linear system from an approximate inverse.

        Returns a belief over the linear system from an approximate inverse
        :math:`H_0\approx A^{-1}` such as a preconditioner.

        Parameters
        ----------
        Ainv0 :
            Approximate inverse of the system matrix.
        problem :
            Linear system to solve.
        """
        if isinstance(Ainv0, (np.ndarray, linops.LinearOperator)):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        A0 = rvs.Normal(mean=problem.A, cov=linops.SymmetricKronecker(A=problem.A))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @classmethod
    def from_matrix(
        cls,
        A0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief over the linear system from an approximate system matrix.

        Returns a belief over the linear system from an approximation of
        the system matrix :math:`A_0\approx A`.

        Parameters
        ----------
        A0 :
            Approximate system matrix.
        problem :
            Linear system to solve.
        """
        if isinstance(A0, (np.ndarray, linops.LinearOperator)):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=problem.A))

        Ainv0_mean = linops.Identity(shape=A0.shape)
        Ainv0 = rvs.Normal(mean=Ainv0_mean, cov=linops.SymmetricKronecker(A=Ainv0_mean))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @classmethod
    def from_matrices(
        cls,
        A0: Union[np.ndarray, rvs.RandomVariable],
        Ainv0: Union[np.ndarray, rvs.RandomVariable],
        problem: LinearSystem,
    ) -> "LinearSystemBelief":
        r"""Construct a belief from an approximate system matrix and
        corresponding inverse.

        Returns a belief over the linear system from an approximation of
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
        """
        if isinstance(A0, (np.ndarray, linops.LinearOperator)):
            A0 = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(A=problem.A))
        if isinstance(Ainv0, (np.ndarray, linops.LinearOperator)):
            Ainv0 = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=Ainv0))

        return cls(
            x=cls._induced_solution_belief(Ainv=Ainv0, b=problem.b),
            Ainv=Ainv0,
            A=A0,
            b=problem.b,
        )

    @staticmethod
    def _induced_solution_cov(
        Ainv: rvs.Normal, b: rvs.RandomVariable
    ) -> linops.LinearOperator:
        r"""Induced covariance of the belief over the solution.

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
        b = rvs.asrandvar(b)
        Wb = Ainv.cov.A @ b.mean
        bWb = (Wb.T @ b.mean).item()

        def _mv(x):
            return 0.5 * (bWb * Ainv.cov.A @ x + Wb @ (Wb.T @ x))

        return linops.LinearOperator(
            shape=Ainv.shape, dtype=float, matvec=_mv, matmat=_mv
        )

    @staticmethod
    def _induced_solution_belief(Ainv: rvs.Normal, b: rvs.RandomVariable) -> rvs.Normal:
        r"""Induced belief over the solution from a belief over the inverse.

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
            cov=LinearSystemBelief._induced_solution_cov(Ainv=Ainv, b=b),
        )

    def optimize_hyperparams(
        self,
        problem: LinearSystem,
        actions: List[np.ndarray],
        observations: List[np.ndarray],
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Optional["probnum.linalg.linearsolvers.LinearSolverState"]:
        """Optimize the hyperparameters of the belief class given observations.

        Parameters
        ----------
        """
        raise NotImplementedError

    def update(
        self,
        problem: LinearSystem,
        observation_op: "probnum.linalg.linearsolvers.observation_ops.ObservationOperator",
        action: np.ndarray,
        observation: np.ndarray,
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Optional["probnum.linalg.linearsolvers.LinearSolverState"]:
        """Update the belief given observations.

        Parameters
        ----------
        """
        if isinstance(observation_op, MatrixMultObservation):
            belief_update = SymMatrixNormalLinearObsBeliefUpdate(
                problem=problem, belief=belief
            )
        else:
            raise NotImplementedError

        (self._x, self._Ainv, self._A, self._b), solver_state = belief_update(
            action=action, observation=observation
        )
