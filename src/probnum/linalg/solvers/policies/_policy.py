"""Base class for policies of probabilistic linear solvers returning actions."""
from typing import Callable, Optional, Tuple

import numpy as np

import probnum  # pylint: disable="unused-import
import probnum.utils
from probnum.problems import LinearSystem
from probnum.type import RandomStateArgType

# Public classes and functions. Order is reflected in documentation.
__all__ = ["Policy"]

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
                "probnum.linalg.solvers.beliefs.LinearSystemBelief",
                RandomStateArgType,
                Optional["probnum.linalg.solvers.LinearSolverState"],
            ],
            "probnum.linalg.solvers.LinearSolverAction",
        ],
        is_deterministic: bool,
        random_state: RandomStateArgType = None,
    ):
        self._policy = policy
        self._is_deterministic = is_deterministic
        self.random_state = probnum.utils.as_random_state(random_state)

    def __call__(
        self,
        problem: LinearSystem,
        belief: "probnum.linalg.solvers.beliefs.LinearSystemBelief",
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> "probnum.linalg.solvers.LinearSolverAction":
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
        """
        return self._policy(problem, belief, self.random_state, solver_state)

    @property
    def is_deterministic(self) -> bool:
        """Is the policy a deterministic function of its arguments or stochastic?"""
        return self._is_deterministic
