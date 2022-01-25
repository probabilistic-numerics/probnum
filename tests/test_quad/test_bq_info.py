"""Basic tests for the BQ info container."""

import numpy as np
import pytest

from probnum.quad import BQIterInfo, BQState, LebesgueMeasure
from probnum.randprocs.kernels import ExpQuad


@pytest.fixture
def info():
    return BQIterInfo()


@pytest.fixture
def bq_state_no_data():
    return BQState(measure=LebesgueMeasure(domain=(0, 1)), kernel=ExpQuad(input_dim=1))


@pytest.fixture
def bq_state():
    nevals = 10
    return BQState(
        measure=LebesgueMeasure(domain=(0, 1)),
        kernel=ExpQuad(input_dim=1),
        nodes=np.zeros([nevals, 1]),
        fun_evals=np.ones(nevals),
    )


def test_info_defaults(info):
    assert info.iteration == 0
    assert info.nevals == 0
    assert not info.has_converged


def test_info_from_iteration(info):
    dnevals = 3

    # iteration and nevals must increase
    new_info = BQIterInfo.from_iteration(info, dnevals=dnevals)
    assert new_info.iteration == info.iteration + 1
    assert new_info.nevals == info.nevals + dnevals
    assert not new_info.has_converged

    # again, since initial iteration and nevals were zero
    new_info = BQIterInfo.from_iteration(new_info, dnevals=dnevals)
    assert new_info.iteration == info.iteration + 2
    assert new_info.nevals == info.nevals + 2 * dnevals
    assert not new_info.has_converged


def test_info_from_bq_state(bq_state, bq_state_no_data):

    # no data given
    new_info = BQIterInfo.from_bq_state(bq_state_no_data)
    assert new_info.iteration == 0
    assert new_info.nevals == 0
    assert not new_info.has_converged

    # nevals must match the data given
    new_info = BQIterInfo.from_bq_state(bq_state)
    assert new_info.iteration == 0
    assert new_info.nevals == bq_state.fun_evals.shape[0]
    assert not new_info.has_converged


def test_info_from_stopping_decision(info):

    new_info = BQIterInfo.from_stopping_decision(info, has_converged=False)
    assert new_info.iteration == info.iteration
    assert new_info.nevals == info.iteration
    assert not new_info.has_converged

    new_info = BQIterInfo.from_stopping_decision(info, has_converged=True)
    assert new_info.iteration == info.iteration
    assert new_info.nevals == info.iteration
    assert new_info.has_converged
