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
def test_solve_accuracy_2norm(
    solver: ProbabilisticLinearSolver,
    prior: beliefs.LinearSystemBelief,
    problem: problems.LinearSystem,
    rng: np.random.Generator,
):
    """Test whether a linear system is solved to sufficient accuracy."""
    belief, _ = solver.solve(prior=prior, problem=problem, rng=rng)
    assert np.linalg.norm(belief.x.mean - problem.solution) < 1e-5


def test_perfect_information(
    solver: ProbabilisticLinearSolver,
    problem: problems.LinearSystem,
):
    """Test whether a solver given perfect information converges instantly."""
    pass
