"""Basic tests for BQ acquisition functions."""

# New policies need to be added to the fixtures 'policy_name' and 'policy_params'
# and 'policy'. Further, add the new policy to the assignment test in
# test_bayesian_quadrature.py.


import numpy as np
import pytest

from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.solvers import BQState
from probnum.quad.solvers.acquisition_functions import WeightedPredictiveVariance
from probnum.randprocs.kernels import ExpQuad


@pytest.fixture(params=[pytest.param(s, id=f"bs{s}") for s in [1, 3, 5]])
def batch_size(request):
    return request.param


@pytest.fixture(params=[pytest.param(a) for a in [WeightedPredictiveVariance]])
def acquisition(request):
    return request.param()


def test_acquisition_shapes(acquisition, input_dim, rng):

    m = LebesgueMeasure(input_dim=input_dim, domain=(0, 1))
    k = ExpQuad(input_shape=(input_dim,))
    bq_state_no_data = BQState(
        measure=m,
        kernel=k,
    )
    nevals = 5
    bq_state = BQState(
        measure=m,
        kernel=k,
        nodes=np.zeros([nevals, input_dim]),
        fun_evals=np.ones(nevals),
    )

    n_nodes = 3
    x = np.ones([n_nodes, input_dim])

    # no data yet in bq_state
    res = acquisition(x, bq_state_no_data)
    assert res[0].shape == (n_nodes,)  # values
    assert res[1] is None  # no gradients yet

    # data has been collected previously
    res = acquisition(x, bq_state)
    assert res[0].shape == (n_nodes,)  # values
    assert res[1] is None  # no gradients yet


def test_acquisition_property_values(acquisition):
    # no gradients yet. this may change (#581).
    assert not acquisition.has_gradients
