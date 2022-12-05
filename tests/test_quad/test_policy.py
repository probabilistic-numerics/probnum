"""Basic tests for BQ policies."""

# New policies need to be added to the fixtures 'policy_name' and 'policy_params'
# and 'policy'. Further, add the new policy to the assignment test in
# test_bayesian_quadrature.py.


import numpy as np
import pytest

from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure
from probnum.quad.solvers import BQState
from probnum.quad.solvers.acquisition_functions import WeightedPredictiveVariance
from probnum.quad.solvers.policies import (
    RandomMaxAcquisitionPolicy,
    RandomPolicy,
    VanDerCorputPolicy,
)
from probnum.randprocs.kernels import ExpQuad


@pytest.fixture(params=[pytest.param(s, id=f"bs{s}") for s in [1, 3, 5]])
def batch_size(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(name, id=name)
        for name in ["RandomPolicy", "VanDerCorputPolicy", "RandomMaxAcquisitionPolicy"]
    ]
)
def policy_name(request):
    return request.param


@pytest.fixture
def policy_params(policy_name, input_dim, batch_size, rng):
    def _get_bq_states(ndim):
        nevals = 5
        bq_state_no_data = BQState(
            measure=LebesgueMeasure(input_dim=ndim, domain=(0, 1)),
            kernel=ExpQuad(input_shape=(ndim,)),
        )
        bq_state = BQState(
            measure=LebesgueMeasure(input_dim=ndim, domain=(0, 1)),
            kernel=ExpQuad(input_shape=(ndim,)),
            nodes=np.zeros([nevals, ndim]),
            fun_evals=np.ones(nevals),
        )
        return bq_state, bq_state_no_data

    params = dict(name=policy_name, input_dim=input_dim)
    params["bq_state"], params["bq_state_no_data"] = _get_bq_states(input_dim)

    if policy_name == "RandomPolicy":
        input_params = dict(
            batch_size=batch_size,
            sample_func=lambda batch_size, rng: np.ones([batch_size, input_dim]),
        )
        params["requires_rng"] = True
    elif policy_name == "VanDerCorputPolicy":
        # Since VanDerCorputPolicy can only produce univariate nodes, this overrides
        # input_dim = 1 for all tests. This is a bit cheap, but pytest parametrization
        # is convoluted enough.
        input_params = dict(
            batch_size=batch_size,
            measure=LebesgueMeasure(input_dim=1, domain=(0, 1)),
        )
        params["bq_state"], params["bq_state_no_data"] = _get_bq_states(1)
        params["input_dim"] = 1
        params["requires_rng"] = False
    elif policy_name == "RandomMaxAcquisitionPolicy":
        # Since RandomMaxAcquisitionPolicy requires batch_size=1, we override it here.
        input_params = dict(
            batch_size=1,
            acquisition_func=WeightedPredictiveVariance(),
            n_candidates=10,
        )
        params["requires_rng"] = True
    else:
        raise NotImplementedError

    params["input_params"] = input_params

    return params


@pytest.fixture()
def policy(policy_params):
    name = policy_params.pop("name")
    input_params = policy_params["input_params"]
    # add new policy to this dict
    policy = dict(
        RandomPolicy=RandomPolicy,
        VanDerCorputPolicy=VanDerCorputPolicy,
        RandomMaxAcquisitionPolicy=RandomMaxAcquisitionPolicy,
    )
    return policy[name](**input_params), policy_params


# Tests shared by all policies start here.


def test_policy_shapes(policy, rng):
    policy, params = policy
    bq_state = params["bq_state"]
    bq_state_no_data = params["bq_state_no_data"]
    input_dim = params["input_dim"]
    batch_size = params["input_params"]["batch_size"]

    # bq state contains data
    assert policy(bq_state, rng).shape == (batch_size, input_dim)

    # bq state contains no data yet
    assert policy(bq_state_no_data, rng).shape == (batch_size, input_dim)


def test_policy_property_values(policy):
    policy, params = policy
    assert policy.requires_rng is params["requires_rng"]


# Tests specific to RandomMaxAcquisitionPolicy start here


def test_random_max_acquisition_attribute_values():
    n_cadidates = 10
    policy = RandomMaxAcquisitionPolicy(
        batch_size=1,
        acquisition_func=WeightedPredictiveVariance(),
        n_candidates=n_cadidates,
    )
    assert policy.n_candidates == n_cadidates


def test_random_max_acquisition_raises():
    # batch size larger than 1
    with pytest.raises(ValueError):
        wrong_batch_size = 2
        RandomMaxAcquisitionPolicy(
            batch_size=wrong_batch_size,
            acquisition_func=WeightedPredictiveVariance(),
            n_candidates=10,
        )

    # n_candidates too small
    with pytest.raises(ValueError):
        wrong_n_candidates = 0
        RandomMaxAcquisitionPolicy(
            batch_size=1,
            acquisition_func=WeightedPredictiveVariance(),
            n_candidates=wrong_n_candidates,
        )


# Tests specific to VanDerCorputPolicy start here


def test_van_der_corput_multi_d_error():
    """Check that van der Corput policy fails in dimensions higher than one."""
    wrong_dimension = 2
    measure = GaussianMeasure(input_dim=wrong_dimension, mean=0.0, cov=1.0)
    with pytest.raises(ValueError):
        VanDerCorputPolicy(1, measure)


@pytest.mark.parametrize("domain", [(-np.Inf, 0), (1, np.Inf), (-np.Inf, np.Inf)])
def test_van_der_corput_infinite_error(domain):
    """Check that van der Corput policy fails on infinite domains."""
    measure = LebesgueMeasure(input_dim=1, domain=domain)
    with pytest.raises(ValueError):
        VanDerCorputPolicy(1, measure)


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128, 256])
def test_van_der_corput_full(n):
    """Test that the full van der Corput sequence is being computed correctly."""
    # Full sequence
    vdc_seq = VanDerCorputPolicy.van_der_corput_sequence(1, n)
    expected_seq = np.linspace(1.0 / n, 1.0 - 1.0 / n, n - 1)
    np.testing.assert_array_equal(np.sort(vdc_seq), expected_seq)


def test_van_der_corput_partial():
    """Test that some partial van der Corput sequences are computed correctly."""
    (n_start, n_end) = (3, 8)
    vdc_seq = VanDerCorputPolicy.van_der_corput_sequence(n_start, n_end)
    expected_seq = np.array([0.75, 0.125, 0.625, 0.375, 0.875])
    np.testing.assert_array_equal(vdc_seq, expected_seq)
    (n_start, n_end) = (4, 11)
    vdc_seq = VanDerCorputPolicy.van_der_corput_sequence(n_start, n_end)
    expected_seq = np.array([0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625, 0.3125])
    np.testing.assert_array_equal(vdc_seq, expected_seq)


def test_van_der_corput_start_value_only():
    """When no end value is given, test if sequence returns the correct value."""
    (n_start, n_end) = (1, 8)
    vdc_seq = VanDerCorputPolicy.van_der_corput_sequence(n_start, n_end)
    vdc_seq_single_value = VanDerCorputPolicy.van_der_corput_sequence(n_end - 1)
    assert vdc_seq_single_value.shape == (1,)
    assert vdc_seq[-1] == vdc_seq_single_value[0]
