"""Tests for a policy returning random unit vectors."""
import pathlib

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import ProbabilisticLinearSolverState, policies

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_policies = case_modules + ".policies"
cases_states = case_modules + ".states"


@parametrize_with_cases("policy", cases=cases_policies, glob="*unit_vector")
@parametrize_with_cases("state", cases=cases_states)
def test_returns_unit_vector(
    policy: policies.LinearSolverPolicy, state: ProbabilisticLinearSolverState
):
    action = policy(state)
    assert np.linalg.norm(action) == pytest.approx(1.0)
