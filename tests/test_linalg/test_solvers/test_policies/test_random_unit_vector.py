"""Tests for a policy returning random unit vectors."""
import pathlib

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from probnum.linalg.solvers import LinearSolverState, policies

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_policies = case_modules + ".policies"
cases_states = case_modules + ".states"


@parametrize_with_cases("policy", cases=cases_policies, glob="*unit_vector*")
@parametrize_with_cases("state", cases=cases_states)
@parametrize("seed", [1, 3, 42])
def test_returns_unit_vector(
    policy: policies.LinearSolverPolicy, state: LinearSolverState, seed: int
):
    rng = np.random.default_rng(seed)
    action = policy(state, rng=rng)
    assert np.linalg.norm(action) == pytest.approx(1.0)


def test_raises_error_for_unsupported_probabilities():
    with pytest.raises(ValueError):
        policies.RandomUnitVectorPolicy(probabilities="not-valid")
