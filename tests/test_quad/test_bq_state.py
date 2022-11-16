"""Basic tests for the BQ info container and BQ state."""

import numpy as np
import pytest

from probnum.quad import IntegrationMeasure, KernelEmbedding, LebesgueMeasure
from probnum.quad.solvers.bq_state import BQIterInfo, BQState
from probnum.randprocs.kernels import ExpQuad, Kernel
from probnum.randvars import Normal


@pytest.fixture
def nevals():
    return 10


@pytest.fixture
def info():
    return BQIterInfo()


@pytest.fixture
def bq_state_no_data():
    return BQState(
        measure=LebesgueMeasure(domain=(0, 1)), kernel=ExpQuad(input_shape=(1,))
    )


@pytest.fixture
def bq_state(nevals):
    return BQState(
        measure=LebesgueMeasure(domain=(0, 1)),
        kernel=ExpQuad(input_shape=(1,)),
        nodes=np.zeros([nevals, 1]),
        fun_evals=np.ones(nevals),
    )


# === BQIterInfo tests start here


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


def test_info_from_bq_state(bq_state, bq_state_no_data, nevals):

    # no evaluations/ data given
    new_info = BQIterInfo.from_bq_state(bq_state_no_data)
    assert new_info.iteration == 0
    assert new_info.nevals == 0
    assert not new_info.has_converged

    # nevals must match the data given
    new_info = BQIterInfo.from_bq_state(bq_state)
    assert new_info.iteration == 0
    assert new_info.nevals == nevals
    assert not new_info.has_converged


def test_info_from_stopping_decision(info):

    # solver converged
    new_info = BQIterInfo.from_stopping_decision(info, has_converged=False)
    assert new_info.iteration == info.iteration
    assert new_info.nevals == info.iteration
    assert not new_info.has_converged

    # solver did not converge converged
    new_info = BQIterInfo.from_stopping_decision(info, has_converged=True)
    assert new_info.iteration == info.iteration
    assert new_info.nevals == info.iteration
    assert new_info.has_converged


# === BQState tests start here


@pytest.mark.parametrize("state", ["bq_state_no_data", "bq_state"])
def test_state_defaults_types(state, request):
    s = request.getfixturevalue(state)

    assert isinstance(s.kernel, Kernel)
    assert isinstance(s.measure, IntegrationMeasure)
    assert isinstance(s.kernel_embedding, KernelEmbedding)
    assert isinstance(s.nodes, np.ndarray)
    assert isinstance(s.fun_evals, np.ndarray)
    assert isinstance(s.gram, np.ndarray)
    assert isinstance(s.kernel_means, np.ndarray)
    assert isinstance(s.previous_integral_beliefs, tuple)
    assert isinstance(s.scale_sq, float)
    assert s.integral_belief is None


@pytest.mark.parametrize("state", ["bq_state_no_data", "bq_state"])
def test_state_defaults_values(state, request):
    s = request.getfixturevalue(state)
    assert s.input_dim == s.measure.input_dim


@pytest.mark.parametrize("state", ["bq_state_no_data", "bq_state"])
def test_state_defaults_shapes(state, request):
    s = request.getfixturevalue(state)
    assert len(s.previous_integral_beliefs) == 0
    assert s.gram.shape == (1, 0)
    assert s.kernel_means.shape == (0,)


def test_state_defaults_node_shapes(bq_state, bq_state_no_data, nevals):

    # no evaluations/ data given
    s = bq_state_no_data

    assert s.nodes.shape == (0, s.input_dim)
    assert s.fun_evals.shape == (0,)

    # evaluations/ data given
    s = bq_state

    assert s.nodes.shape == (nevals, s.input_dim)
    assert s.fun_evals.shape == (nevals,)


@pytest.mark.parametrize("state", ["bq_state_no_data", "bq_state"])
def test_state_from_new_data(state, request):

    old_state = request.getfixturevalue(state)
    new_nevals = 5

    # some new data
    x = np.zeros([new_nevals, old_state.input_dim])
    y = np.ones(new_nevals)
    integral = Normal(0, 1)
    gram = np.eye(new_nevals)
    kernel_means = np.ones(new_nevals)
    kernel = ExpQuad(input_shape=(old_state.input_dim,))
    scale_sq = 1.7

    # previously no data given
    s = BQState.from_new_data(
        kernel=kernel,
        scale_sq=scale_sq,
        nodes=x,
        fun_evals=y,
        integral_belief=integral,
        prev_state=old_state,
        gram=gram,
        kernel_means=kernel_means,
    )

    # types
    assert isinstance(s.kernel, Kernel)
    assert isinstance(s.measure, IntegrationMeasure)
    assert isinstance(s.kernel_embedding, KernelEmbedding)
    assert isinstance(s.nodes, np.ndarray)
    assert isinstance(s.fun_evals, np.ndarray)
    assert isinstance(s.gram, np.ndarray)
    assert isinstance(s.kernel_means, np.ndarray)
    assert isinstance(s.integral_belief, Normal)
    assert isinstance(s.previous_integral_beliefs, tuple)

    # shapes
    assert s.nodes.shape == (new_nevals, s.input_dim)
    assert s.fun_evals.shape == (new_nevals,)
    assert len(s.previous_integral_beliefs) == 1
    assert s.gram.shape == (new_nevals, new_nevals)
    assert s.kernel_means.shape == (new_nevals,)

    # values
    assert s.input_dim == s.measure.input_dim
    assert s.scale_sq == scale_sq
