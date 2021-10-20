"""Tests for the matrix-vector product information operator."""

import pathlib

import numpy as np
from pytest_cases import parametrize_with_cases

from probnum.linalg.solvers import LinearSolverState, information_ops

case_modules = (pathlib.Path(__file__).parent / "cases").stem
cases_information_ops = case_modules + ".information_ops"
cases_states = case_modules + ".states"


@parametrize_with_cases("info_op", cases=cases_information_ops, glob="*matvec")
@parametrize_with_cases("state", cases=cases_states, has_tag=["has_action"])
def test_is_A_matvec(
    info_op: information_ops.LinearSolverInfoOp, state: LinearSolverState
):
    observation = info_op(state)
    np.testing.assert_equal(observation, state.problem.A @ state.action)
