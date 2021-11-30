"""Tests for probabilistic linear solvers applied to general linear systems."""

import pathlib

import numpy as np
from pytest_cases import filters, parametrize_with_cases

from probnum import linops, problems, randvars
from probnum.linalg.solvers import ProbabilisticLinearSolver, beliefs

case_modules = pathlib.Path("cases").stem
cases_solvers = case_modules + ".solvers"
cases_beliefs = case_modules + ".beliefs"
cases_problems = case_modules + ".problems"


@parametrize_with_cases("solver", cases=cases_solvers, filter=~filters.has_tag("sym"))
@parametrize_with_cases("prior", cases=cases_beliefs, filter=~filters.has_tag("sym"))
@parametrize_with_cases("problem", cases=cases_problems)
def test_small_residual(
    solver: ProbabilisticLinearSolver,
    prior: beliefs.LinearSystemBelief,
    problem: problems.LinearSystem,
):
    """Test whether the output solution has small residual."""
    belief, _ = solver.solve(
        prior=prior, problem=problem, rng=np.random.default_rng(42)
    )

    residual_norm = np.linalg.norm(problem.A @ belief.x.mean - problem.b, ord=2)

    assert residual_norm < 1e-5 or residual_norm < 1e-5 * np.linalg.norm(
        problem.b, ord=2
    )


@parametrize_with_cases("solver", cases=cases_solvers, filter=~filters.has_tag("sym"))
@parametrize_with_cases("problem", cases=cases_problems)
def test_perfect_information(
    solver: ProbabilisticLinearSolver, problem: problems.LinearSystem, ncols: int
):
    """Test whether a solver given perfect information converges instantly."""

    # Construct prior belief with perfect information
    belief = beliefs.LinearSystemBelief(
        x=randvars.Normal(
            mean=problem.solution, cov=linops.Scaling(factors=0.0, shape=(ncols, ncols))
        ),
        A=randvars.Constant(problem.A),
        Ainv=randvars.Constant(np.linalg.inv(problem.A @ np.eye(ncols))),
    )

    # Run solver
    belief, solver_state = solver.solve(
        prior=belief, problem=problem, rng=np.random.default_rng(1)
    )

    # Check for instant convergence
    assert solver_state.step == 0
    np.testing.assert_allclose(belief.x.mean, problem.solution)
