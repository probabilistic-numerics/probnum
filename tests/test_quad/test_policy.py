"""Basic tests for BQ policies."""


# New policies need to be added to the fixtures 'policy_name' and 'policy_params'
# and 'policy'.


import numpy as np
import pytest

from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure
from probnum.quad.solvers import BQState
from probnum.quad.solvers.policies import RandomPolicy, VanDerCorputPolicy
from probnum.randprocs.kernels import ExpQuad


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture(
    params=[
        pytest.param(name, id=name) for name in ["RandomPolicy", "VanDerCorputPolicy"]
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

    params = dict(name=policy_name, ndim=input_dim)
    params["bq_state"], params["bq_state_no_data"] = _get_bq_states(input_dim)

    if policy_name == "RandomPolicy":
        input_params = dict(
            batch_size=batch_size,
            sample_func=lambda batch_size, rng: np.ones([batch_size, input_dim]),
        )
    elif policy_name == "VanDerCorputPolicy":
        # Since VanDerCorputPolicy can only produce univariate nodes, this overrides
        # input_dim = 1 for all tests. This is a bit cheap, but pytest parametrization
        # is convoluted enough.
        input_params = dict(
            batch_size=batch_size,
            measure=LebesgueMeasure(input_dim=1, domain=(0, 1)),
        )
        params["bq_state"], params["bq_state_no_data"] = _get_bq_states(1)
        params["ndim"] = 1
    else:
        raise NotImplementedError

    params["input_params"] = input_params

    return params


@pytest.fixture()
def policy(policy_params):
    name = policy_params.pop("name")
    input_params = policy_params.pop("input_params")

    if name == "RandomPolicy":
        return RandomPolicy(**input_params), policy_params
    elif name == "VanDerCorputPolicy":
        return VanDerCorputPolicy(**input_params), policy_params
    else:
        raise NotImplementedError


# Tests shared by all policies start here.


def test_policy_shapes(policy, batch_size, rng):
    policy, params = policy
    bq_state, bq_state_no_data = params["bq_state"], params["bq_state_no_data"]
    ndim = params["ndim"]

    # bq state contains data
    assert policy(bq_state=bq_state, rng=rng).shape == (batch_size, ndim)

    # bq state contains no data yet
    assert policy(bq_state=bq_state_no_data, rng=rng).shape == (batch_size, ndim)


# Tests specific to VanDerCorputPolicy start here


def test_van_der_corput_multi_d_error():
    """Check that van der Corput policy fails in dimensions higher than one."""
    wrong_dimension = 2
    measure = GaussianMeasure(input_dim=wrong_dimension, mean=0.0, cov=1.0)
    with pytest.raises(ValueError):
        VanDerCorputPolicy(measure, batch_size=1)


@pytest.mark.parametrize("domain", [(-np.Inf, 0), (1, np.Inf), (-np.Inf, np.Inf)])
def test_van_der_corput_infinite_error(domain):
    """Check that van der Corput policy fails on infinite domains."""
    measure = LebesgueMeasure(input_dim=1, domain=domain)
    with pytest.raises(ValueError):
        VanDerCorputPolicy(measure, batch_size=1)


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
