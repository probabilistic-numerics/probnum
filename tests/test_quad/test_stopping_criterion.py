"""Tests for BQ stopping criteria."""

from typing import Tuple

import numpy as np
import pytest

from probnum.quad import (
    BQStoppingCriterion,
    ImmediateStop,
    IntegralVarianceTolerance,
    LebesgueMeasure,
    MaxNevals,
    RelativeMeanChange,
)
from probnum.quad.solvers.bq_state import BQIterInfo, BQState
from probnum.randprocs.kernels import ExpQuad
from probnum.randvars import Normal

_nevals = 5
_rel_tol = 1e-5
_var_tol = 1e-5


@pytest.fixture()
def input_dim() -> int:
    return 2


@pytest.fixture(
    params=[
        pytest.param(sc, id=sc[0].__name__)
        for sc in [
            (MaxNevals, {"max_nevals": _nevals}),
            (IntegralVarianceTolerance, {"var_tol": _var_tol}),
            (RelativeMeanChange, {"rel_tol": _rel_tol}),
        ]
    ],
    name="stopping_criterion",
)
def fixture_stopping_criterion(request) -> BQStoppingCriterion:
    """BQ stopping criterion."""
    return request.param[0](**request.param[1])


@pytest.fixture()
def bq_state_stops(input_dim) -> Tuple[BQState, BQIterInfo]:
    """BQ state that triggers stopping in all stopping criteria."""
    integral_mean = 1.0
    integral_mean_previous = integral_mean * (1 - _rel_tol)

    bq_state = BQState(
        measure=LebesgueMeasure(input_dim=input_dim, domain=(0, 1)),
        kernel=ExpQuad(input_shape=(input_dim,)),
        integral_belief=Normal(integral_mean, 0.1 * _var_tol),
        previous_integral_beliefs=(Normal(integral_mean_previous, _var_tol),),
        nodes=np.ones((_nevals, input_dim)),
        fun_evals=np.ones(_nevals),
    )
    info = BQIterInfo.from_bq_state(bq_state)
    return bq_state, info


@pytest.fixture()
def bq_state_does_not_stop(input_dim) -> Tuple[BQState, BQIterInfo]:
    """BQ state that does not trigger stopping in all stopping criteria."""
    integral_mean = 1.0
    integral_mean_previous = 2 * integral_mean * (1 - _rel_tol)
    nevals = _nevals - 2
    bq_state = BQState(
        measure=LebesgueMeasure(input_dim=input_dim, domain=(0, 1)),
        kernel=ExpQuad(input_shape=(input_dim,)),
        integral_belief=Normal(integral_mean, 10 * _var_tol),
        previous_integral_beliefs=(Normal(integral_mean_previous, _var_tol),),
        nodes=np.ones((nevals, input_dim)),
        fun_evals=np.ones(nevals),
    )
    info = BQIterInfo.from_bq_state(bq_state)
    return bq_state, info


def test_immediate_stop_values(bq_state_stops, bq_state_does_not_stop):
    # Immediate stop shall always stop
    sc = ImmediateStop()

    state, info = bq_state_stops
    assert sc(state, info)

    state, info = bq_state_does_not_stop
    assert sc(state, info)


def test_stopping_criterion_values(
    stopping_criterion, bq_state_stops, bq_state_does_not_stop
):
    state, info = bq_state_stops
    assert stopping_criterion(state, info)

    state, info = bq_state_does_not_stop
    assert not stopping_criterion(state, info)
