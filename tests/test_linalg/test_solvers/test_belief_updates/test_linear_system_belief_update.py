"""Tests for belief updates about quantities of interest of a linear system."""

import pathlib

from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import LinearSolverState, belief_updates, beliefs

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_belief_updates = case_modules + ".belief_updates"
cases_states = case_modules + ".states"


@parametrize_with_cases("belief_update", cases=cases_belief_updates)
@parametrize_with_cases("state", cases=cases_states)
def test_returns_linear_system_belief(
    belief_update: belief_updates.LinearSystemBeliefUpdate, state: LinearSolverState
):
    belief = belief_update(solver_state=state)
    assert isinstance(belief, beliefs.LinearSystemBelief)
