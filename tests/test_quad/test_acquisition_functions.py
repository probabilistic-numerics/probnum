"""Basic tests for BQ acquisition functions."""

# New acquisition functions need to be added to the fixtures 'acquisition'.


import numpy as np
import pytest

from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.solvers import BQState
from probnum.quad.solvers.acquisition_functions import WeightedPredictiveVariance
from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate
from probnum.randprocs.kernels import ExpQuad
from probnum.randvars import Normal


@pytest.fixture(params=[pytest.param(n, id=f"nevals{n}") for n in [1, 3, 5]])
def nevals(request):
    return request.param


@pytest.fixture
def belief_update():
    return BQStandardBeliefUpdate(jitter=1e-8, scale_estimation=None)


@pytest.fixture(params=[pytest.param(a) for a in [WeightedPredictiveVariance]])
def acquisition(request):
    return request.param()


def test_acquisition_shapes(acquisition, input_dim, nevals, rng, belief_update):

    m = LebesgueMeasure(input_dim=input_dim, domain=(0, 1))
    k = ExpQuad(input_shape=(input_dim,))
    bq_state_no_data = BQState(
        measure=m,
        kernel=k,
    )
    bq_state = BQState(
        measure=m,
        kernel=k,
        nodes=np.zeros([nevals, input_dim]),
        fun_evals=np.ones(nevals),
        gram=np.eye(nevals),
        kernel_means=np.ones(nevals),
        previous_integral_beliefs=tuple(
            [Normal(mean=0.0, cov=1.0) for _ in range(nevals)]
        ),
        integral_belief=Normal(mean=0.0, cov=1.0),
    )

    n_nodes = 3
    x = np.ones([n_nodes, input_dim])

    # no data yet in bq_state
    res = acquisition(x, bq_state_no_data, belief_update)
    assert res[0].shape == (n_nodes,)  # values
    assert res[1] is None  # no gradients yet

    # data has been collected previously
    res = acquisition(x, bq_state, belief_update)
    assert res[0].shape == (n_nodes,)  # values
    assert res[1] is None  # no gradients yet


def test_acquisition_property_values(acquisition):
    # no gradients yet. this may change (#581).
    assert not acquisition.has_gradients
