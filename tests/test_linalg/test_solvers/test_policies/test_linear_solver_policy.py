"""Tests for probabilistic linear solver policies."""
import pathlib

import numpy as np
from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import ProbabilisticLinearSolverState, policies

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_policies = case_modules + ".policies"
cases_states = case_modules + ".states"


@parametrize_with_cases("policy", cases=cases_policies)
@parametrize_with_cases("state", cases=cases_states)
def test_returns_ndarray(
    policy: policies.LinearSolverPolicy, state: ProbabilisticLinearSolverState
):
    action = policy(state)
    assert isinstance(action, np.ndarray)


@parametrize_with_cases("policy", cases=cases_policies)
@parametrize_with_cases("state", cases=cases_states)
def test_shape(
    policy: policies.LinearSolverPolicy, state: ProbabilisticLinearSolverState
):
    action = policy(state)
    assert action.shape[0] == state.problem.A.shape[1]


@parametrize_with_cases("policy", cases=cases_policies, has_tag="random")
@parametrize_with_cases("state", cases=cases_states)
def test_uses_solver_state_random_number_generator(
    policy: policies.LinearSolverPolicy, state: ProbabilisticLinearSolverState
):
    """Test whether randomized policies make use of the random number generator stored
    in the linear solver state."""
    rng_state_pre = state.rng.bit_generator.state["state"]["state"]
    _ = policy(state)
    rng_state_post = state.rng.bit_generator.state["state"]["state"]
    assert rng_state_pre != rng_state_post
