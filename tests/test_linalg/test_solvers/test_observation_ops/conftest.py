"""Test fixtures for observation operators."""

import pytest

from probnum.linalg.solvers import observation_ops


@pytest.fixture(
    params=[
        pytest.param(observation_op, id=observation_op_name)
        for (observation_op_name, observation_op) in zip(
            ["matvec"],
            [observation_ops.MatVec()],
        )
    ],
    name="observation_op",
)
def fixture_observation_op(request) -> observation_ops.ObservationOperator:
    """Observation operators of linear solvers."""
    return request.param
