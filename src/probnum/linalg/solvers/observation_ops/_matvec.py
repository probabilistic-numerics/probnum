"""Matrix-vector product observations of a linear system."""
from typing import Optional, Tuple

import probnum
import probnum.random_variables as rvs
from probnum.linalg.solvers.data import LinearSolverAction, LinearSolverObservation
from probnum.linalg.solvers.observation_ops._observation_operator import (
    ObservationOperator,
)
from probnum.problems import LinearSystem

# pylint: disable="invalid-name"

# Public classes and functions. Order is reflected in documentation.
__all__ = ["MatVecObservation"]


class MatVecObservation(ObservationOperator):
    r"""Matrix-vector product observations.

    Given an action :math:`s` collect an observation :math:`y` of the linear system by
    multiplying with the system matrix :math:`y = As`.
    """

    def __init__(self):
        super().__init__(observation_op=self.__call__)

    def __call__(
        self,
        problem: LinearSystem,
        action: LinearSolverAction,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> LinearSolverObservation:

        if isinstance(problem.A, rvs.RandomVariable):
            obs_A = problem.A.sample()
        else:
            obs_A = problem.A

        if isinstance(problem.b, rvs.RandomVariable):
            obs_b = problem.b.sample()
        else:
            obs_b = problem.b

        return LinearSolverObservation(A=obs_A @ action.A, b=obs_b)
