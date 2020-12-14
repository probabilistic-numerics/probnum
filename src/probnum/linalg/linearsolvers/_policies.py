"""Policies of probabilistic linear solvers returning actions."""
from typing import Tuple

import numpy as np

import probnum
import probnum.random_variables as rvs
from probnum.type import RandomStateArgType


def conjugate_directions_policy(
    problem: "probnum.problems.LinearSystem",
    belief: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
    random_state: RandomStateArgType = None,
) -> np.ndarray:
    """Policy returning A-conjugate directions.

    Parameters
    ----------
    problem :
        Linear system to solve.
    belief :
        Belief over the parameters of the linear system.
    random_state :
        Random state of the policy. If None (or np.random), the global np.random state
        is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """
    x, _, Ainv = belief
    return Ainv @ (problem.A @ x - problem.b)


def explore_exploit_policy(
    problem: "probnum.problems.LinearSystem",
    belief: Tuple[rvs.RandomVariable, rvs.RandomVariable, rvs.RandomVariable],
    random_state: RandomStateArgType = None,
) -> np.ndarray:
    """Policy trading off exploration and exploitation.

    Parameters
    ----------
    problem :
        Linear system to solve.
    belief :
        Belief over the parameters of the linear system.
    random_state :
        Random state of the policy. If None (or np.random), the global np.random state
        is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    """
    return NotImplementedError
