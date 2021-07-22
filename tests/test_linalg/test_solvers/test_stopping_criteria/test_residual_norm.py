"""Tests for the residual norm stopping criterion."""

import pathlib

from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import ProbabilisticLinearSolverState, stopping_criteria

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_stopping_criteria = case_modules + ".stopping_criteria"
cases_states = case_modules + ".states"


@parametrize_with_cases(
    "stop_crit", cases=cases_stopping_criteria, glob="*residual_norm"
)
@parametrize_with_cases("state", cases=cases_states, glob="*converged")
def test_has_converged(
    stop_crit: stopping_criteria.LinearSolverStoppingCriterion,
    state: ProbabilisticLinearSolverState,
):
    assert stop_crit(solver_state=state)
