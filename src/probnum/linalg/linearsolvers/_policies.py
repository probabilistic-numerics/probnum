"""Policies of probabilistic linear solvers returning actions."""
from typing import Callable, Optional, Tuple

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

    See Also
    --------
    ConjugateDirectionsPolicy : Policy returning :math:`A`-conjugate actions.
    ExploreExploitPolicy: Policy trading off exploration and exploitation.
    """

    def __init__(
        self,
        policy: Callable[
            [
                LinearSystem,
                "probnum.linalg.linearsolvers.LinearSystemBelief",
                RandomStateArgType,
                Optional["probnum.linalg.linearsolvers.LinearSolverState"],
            ],
            Tuple[
                np.ndarray, Optional["probnum.linalg.linearsolvers.LinearSolverState"]
            ],
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
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[np.ndarray, Optional["probnum.linalg.linearsolvers.LinearSolverState"]]:
        """Return an action based on the given problem and model.

        Parameters
        ----------
        problem :
            Linear system to solve.
        belief
            Belief over the solution :math:`x`, the system matrix :math:`A`, its
            inverse :math:`H=A^{-1}` and the right hand side :math:`b`.
        solver_state :
            Current state of the linear solver.

        Returns
        -------
        action :
            Action chosen by the policy.
        solver_state :
            Updated solver state.
        """
        return self._policy(problem, belief, self.random_state, solver_state)

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
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[np.ndarray, Optional["probnum.linalg.linearsolvers.LinearSolverState"]]:
        # Residual
        if solver_state.residual is None:
            solver_state.residual = problem.A @ belief.x.mean - problem.b

        # A-conjugate search direction / action (assuming exact arithmetic)
        action = -belief.Ainv.mean @ solver_state.residual
        solver_state.actions.append(action)
        return action, solver_state


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
        belief: "probnum.linalg.linearsolvers.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.linearsolvers.LinearSolverState"] = None,
    ) -> Tuple[np.ndarray, Optional["probnum.linalg.linearsolvers.LinearSolverState"]]:

        # Residual
        if solver_state.residual is None:
            solver_state.residual = problem.A @ belief.x.mean - problem.b

        # Explore - exploit action
        action = rvs.Normal(
            -belief.Ainv.mean @ solver_state.residual, belief.x.cov
        ).sample()
        solver_state.actions.append(action)
        return action, solver_state
