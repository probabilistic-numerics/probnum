"""Tests for probabilistic linear solvers."""
import pathlib

import numpy as np
from pytest_cases import parametrize_with_cases

from probnum import problems
from probnum.linalg.solvers import ProbabilisticLinearSolver, beliefs

case_modules = pathlib.Path("cases").stem
cases_solvers = case_modules + ".solvers"
cases_beliefs = case_modules + ".beliefs"
cases_problems = case_modules + ".problems"


@parametrize_with_cases("solver", cases=cases_solvers)
@parametrize_with_cases("prior", cases=cases_beliefs)
@parametrize_with_cases("problem", cases=cases_problems)
def test_small_residual(
    solver: ProbabilisticLinearSolver,
    prior: beliefs.LinearSystemBelief,
    problem: problems.LinearSystem,
):
    """Test whether the output solution has small residual."""
    belief, solver_state = solver.solve(
        prior=prior, problem=problem, rng=np.random.default_rng(42)
    )

    residual_norm = np.linalg.norm(problem.A @ belief.x.mean - problem.b, ord=2)

    assert residual_norm < 1e-5 or residual_norm < 1e-5 * np.linalg.norm(
        problem.b, ord=2
    )


def test_perfect_information(
    solver: ProbabilisticLinearSolver,
    problem: problems.LinearSystem,
):
    """Test whether a solver given perfect information converges instantly."""
    pass
