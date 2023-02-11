"""Basic tests for BQ acquisition functions."""

# New acquisition functions need to be added to the fixtures 'acquisition'.


import numpy as np
import pytest

from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.solvers import BQState
from probnum.quad.solvers.acquisition_functions import (
    IntegralVarianceReduction,
    MutualInformation,
    WeightedPredictiveVariance,
)
from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate
from probnum.randprocs.kernels import ExpQuad
from probnum.randvars import Normal


@pytest.fixture(params=[pytest.param(n, id=f"nevals{n}") for n in [0, 1, 3, 5]])
def nevals(request):
    # 0 means no data
    return request.param


@pytest.fixture(
    params=[
        pytest.param(a)
        for a in [
            IntegralVarianceReduction,
            MutualInformation,
            WeightedPredictiveVariance,
        ]
    ]
)
def acquisition(request):
    return request.param()


@pytest.fixture
def bq_state(input_dim, nevals, rng):

    # no data if nevals == 0
    measure = LebesgueMeasure(input_dim=input_dim, domain=(0, 1))
    kernel = ExpQuad(input_shape=(input_dim,), lengthscales=0.1)
    integral = Normal(0.0, KernelEmbedding(kernel, measure).kernel_variance())
    bq_state = BQState(measure=measure, kernel=kernel, integral_belief=integral)

    # with data
    if nevals > 0:
        nodes = rng.uniform(size=(nevals, input_dim))
        fun_evals = np.sin(10 * np.linalg.norm(nodes, axis=1))
        belief_update = BQStandardBeliefUpdate(jitter=1e-8, scale_estimation=None)
        _, bq_state = belief_update(
            bq_state=bq_state,
            new_nodes=nodes,
            new_fun_evals=fun_evals,
        )
    return bq_state


@pytest.mark.parametrize("n_pred", [1, 7, 9], ids=[f"npred{n}" for n in [1, 7, 9]])
def test_acquisition_shapes(acquisition, n_pred, bq_state, rng):
    x = bq_state.measure.sample(n_sample=n_pred, rng=rng)

    res = acquisition(x, bq_state)
    assert res[0].shape == (n_pred,)  # values
    assert res[1] is None  # no gradients yet


def test_acquisition_property_values(acquisition):
    # no gradients yet. this may change (#581).
    assert not acquisition.has_gradients
