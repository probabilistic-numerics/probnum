"""Tests for the maximum iterations stopping criterion."""

import pathlib

from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import ProbabilisticLinearSolverState, stopping_criteria

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_stopping_criteria = case_modules + ".stopping_criteria"
cases_states = case_modules + ".states"


@parametrize_with_cases("state", cases=cases_states, glob="*initial_state")
def test_maxiter_None(state: ProbabilisticLinearSolverState):
    """Test whether if ``maxiter=None``, the maximum number of iterations is set to
    :math:`10n`, where :math:`n` is the dimension of the linear system."""
    stop_crit = stopping_criteria.MaxIterationsStopCrit()

    for _ in range(10 * state.problem.A.shape[1]):
        assert not stop_crit(state)
        state.next_step()

    assert stop_crit(state)
