"""Matrix-vector product observations of a linear system."""
from typing import Optional, Tuple

import probnum
from probnum.linalg.solvers.data import LinearSolverAction, LinearSolverObservation
from probnum.linalg.solvers.observation_ops._observation_operator import ObservationOp
from probnum.problems import LinearSystem, NoisyLinearSystem

# pylint: disable="invalid-name"

# Public classes and functions. Order is reflected in documentation.
__all__ = ["MatVec", "SampleMatVec"]


class MatVec(ObservationOp):
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

        return LinearSolverObservation(
            obsA=problem.A @ action.actA, obsb=problem.b, A=problem.A, b=problem.b
        )


class SampleMatVec(ObservationOp):
    r"""Matrix-vector product observation of a noisy linear system.

    Observe a noisy linear system by drawing a sample linear problem :math:`(A +E,
    b + \epsilon)` and multiplying with the system matrix :math:`y=(A +E) s`.
    """

    def __init__(self):
        super().__init__(observation_op=self.__call__)

    def __call__(
        self,
        problem: NoisyLinearSystem,
        action: LinearSolverAction,
        solver_state: Optional["probnum.linalg.solvers.LinearSolverState"] = None,
    ) -> LinearSolverObservation:

        A, b = problem.sample()

        return LinearSolverObservation(obsA=A @ action.actA, obsb=b, A=A, b=b)
