"""Probabilistic linear solver state test cases."""

from probnum import linalg, problems


def case_linear_solver_state(
    linsys: problems.LinearSystem, prior: linalg.solvers.beliefs.LinearSystemBelief
):
    """State of a linear solver."""
    return linalg.solvers.LinearSolverState(problem=linsys, prior=prior)
