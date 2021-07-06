"""Probabilistic linear solver state test cases."""

from pytest_cases import parametrize_with_cases

from probnum import linalg, problems


@parametrize_with_cases("linsys", cases=None)
@parametrize_with_cases("prior", cases=None)
def case_linear_solver_state(
    linsys: problems.LinearSystem, prior: linalg.solvers.beliefs.LinearSystemBelief
):
    """State of a linear solver."""
    return linalg.solvers.LinearSolverState(problem=linsys, prior=prior)
