"""Test cases for the BQ belief updater."""

import numpy as np
import pytest

from probnum.quad.integration_measures import LebesgueMeasure
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.solvers import BQState
from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate
from probnum.randprocs.kernels import ExpQuad
from probnum.randvars import Normal


@pytest.fixture(params=[pytest.param(n, id=f"nevals{n}") for n in [0, 1, 5]])
def nevals(request) -> BQState:
    # 0 means no data
    return request.param


@pytest.fixture
def bq_state(input_dim, nevals, rng):
    measure = LebesgueMeasure(input_dim=input_dim, domain=(0, 1))
    kernel = ExpQuad(input_shape=(input_dim,), lengthscales=0.1)
    integral = Normal(0.0, KernelEmbedding(kernel, measure).kernel_variance())
    bq_state = BQState(measure=measure, kernel=kernel, integral_belief=integral)

    # update BQ state if nodes and evaluations are available
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


@pytest.mark.parametrize(
    "n_nodes_predict", [1, 2, 3, 5], ids=[f"npred{n}" for n in [1, 2, 3, 5]]
)
def test_predict_integrand_shapes(bq_state, n_nodes_predict):
    x = np.random.rand(n_nodes_predict, bq_state.input_dim)
    res = BQStandardBeliefUpdate.predict_integrand(x, bq_state)
    assert res[0].shape == (n_nodes_predict,)
    assert res[1].shape == (n_nodes_predict,)


def test_belief_update_raises():
    # negative jitter is not allowed
    wrong_jitter = -1.0
    with pytest.raises(ValueError):
        BQStandardBeliefUpdate(jitter=wrong_jitter, scale_estimation="mle")
