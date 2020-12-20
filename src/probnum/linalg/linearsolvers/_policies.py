"""Policies of probabilistic linear solvers returning actions."""
from typing import Callable

import numpy as np

import probnum  # pylint: disable="unused-import
import probnum.random_variables as rvs
from probnum.problems import LinearSystem
from probnum.type import RandomStateArgType

# pylint: disable="invalid-name"


class Policy:
    """Policy of a (probabilistic) linear solver.

    The policy :math:`\\pi(s \\mid \\mathsf{A}, \\mathsf{H}, \\mathsf{x}, A, b)` of a
    linear solver returns a vector to probe the linear system with, typically via
    multiplication, resulting in an observation. Policies can either be deterministic or
    stochastic depending on the application. In the quadratic optimization view of
    solving linear systems the actions :math:`s` are the search directions of the
    optimizer.

    Parameters
    ----------
    policy
        Callable defining the policy and returning an action.
    is_deterministic
        Is the policy a deterministic function of its arguments or stochastic (i.e.
        sampling-based)?
    random_state
        Random state of the policy. If None (or :mod:`numpy.random`), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """

    def __init__(
        self,
        policy: Callable[
            [
                LinearSystem,
                "probnum.linalg.linearsolvers.LinearSolverState",
                RandomStateArgType,
            ],
            np.ndarray,
        ],
        is_deterministic: bool,
        random_state: RandomStateArgType = None,
    ):
        self._policy = policy
        self._is_deterministic = is_deterministic
        self.random_state = random_state

    def __call__(
        self,
        problem: LinearSystem,
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> np.ndarray:
        """Return an action based on the given problem and model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        solver_state :
            Current state of the linear solver.
        """
        return self._policy(problem, solver_state, self.random_state)

    @property
    def is_deterministic(self) -> bool:
        """Is the policy a deterministic function of its arguments or stochastic?"""
        return self._is_deterministic


class ConjugateDirectionsPolicy(Policy):
    """Policy returning :math:`A`-conjugate directions.

    Returns an action given by :math:`s_i = -\\mathbb{E}[\\mathsf{H}]r_{i-1}` where
    :math:`r_{i-1} = A x_{i-1} - b` is the current residual. If the posterior mean of
    :math:`\\mathbb{E}[\\mathsf{H}]` of the inverse model equals the true inverse,
    the resulting action is the exact step to the solution of the linear system. [1]_

    References
    ----------
    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
       *Advances in Neural Information Processing Systems (NeurIPS)*, 2020
    """

    def __init__(self):
        super().__init__(policy=self.__call__, is_deterministic=True)

    def __call__(
        self,
        problem: LinearSystem,
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> np.ndarray:
        x, _, Ainv, _ = solver_state.belief
        residual = solver_state.residual
        if residual is None:
            residual = problem.A @ x.mean - problem.b
        return -Ainv.mean @ residual


class ExploreExploitPolicy(Policy):
    """Policy trading off exploration and exploitation.

    Returns an action given by :math:`s_i \\sim \\mathcal{N}(s; -\\mathbb{E}[
    \\mathsf{H}]r_{i-1}, \\mathbb{Cov}(\\mathsf{x})))` where :math:`r_{i-1} = A x_{i-1}
    - b` is the current residual and :math:`\\mathbb{Cov}(\\mathsf{x})` the
    uncertainty of the solution estimate.

    Parameters
    ----------
    random_state
        Random state of the policy. If None (or :mod:`numpy.random`), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """

    def __init__(self, random_state=None):
        super().__init__(
            policy=self.__call__, is_deterministic=False, random_state=random_state
        )

    def __call__(
        self,
        problem: LinearSystem,
        solver_state: "probnum.linalg.linearsolvers.LinearSolverState",
    ) -> np.ndarray:
        x, _, Ainv, _ = solver_state.belief
        return rvs.Normal(-Ainv.mean @ solver_state.residual, x.cov).sample()
