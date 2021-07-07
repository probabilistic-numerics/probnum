"""Tests for a policy returning random unit vectors."""
import pathlib

from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import LinearSolverState, policies

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_policies = case_modules + ".policies"
cases_states = case_modules + ".states"


@parametrize_with_cases("policy", cases=cases_policies, glob="*conjugate_actions")
@parametrize_with_cases("state", cases=cases_states)
def test_conjugate_actions(
    policy: policies.LinearSolverPolicy, state: LinearSolverState
):
    pass
