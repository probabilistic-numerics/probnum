"""Basic tests for BQ policies."""

import numpy as np
import pytest

from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure
from probnum.quad.solvers import BQState
from probnum.quad.solvers.policies import Policy, RandomPolicy, VanDerCorputPolicy
from probnum.randprocs.kernels import ExpQuad


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture
def bq_state_no_data(input_dim):
    return BQState(
        measure=LebesgueMeasure(input_dim=input_dim, domain=(0, 1)),
        kernel=ExpQuad(input_shape=(input_dim,)),
    )


@pytest.fixture
def bq_state(input_dim):
    nevals = 5
    return BQState(
        measure=LebesgueMeasure(input_dim=input_dim, domain=(0, 1)),
        kernel=ExpQuad(input_shape=(input_dim,)),
        nodes=np.zeros([nevals, input_dim]),
        fun_evals=np.ones(nevals),
    )


@pytest.fixture
def sample_func(batch_size, input_dim, rng):
    def f(batch_size, rng):
        return np.ones([batch_size, input_dim])

    return f


@pytest.fixture(
    params=[
        pytest.param(sc, id=sc[0].__name__)
        for sc in [
            (RandomPolicy, dict(batch_size="batch_size", sample_func="sample_func")),
        ]
    ],
    name="policy",
)
def fixture_policy(request) -> Policy:
    """Policies that only allow univariate inputs need to be handled separately."""
    params = {}
    for key in request.param[1]:
        params[key] = request.getfixturevalue(request.param[1][key])
    return request.param[0](**params)


def test_policy_shapes(policy, batch_size, rng, input_dim, bq_state, bq_state_no_data):

    # bq state contains data
    assert policy(bq_state=bq_state, rng=rng).shape == (batch_size, input_dim)

    # bq state contains no data yet
    assert policy(bq_state=bq_state_no_data, rng=rng).shape == (batch_size, input_dim)


# Tests specific to VanDerCorputPolicy start here


def test_van_der_corput_shapes(batch_size, rng):
    """This is the same test as test_policies_shapes but for 1d only."""
    measure = LebesgueMeasure(domain=(0, 1))
    policy = VanDerCorputPolicy(measure=measure, batch_size=batch_size)

    # bq state contains no data yet
    bq_state_no_data = BQState(measure=measure, kernel=ExpQuad(input_shape=(1,)))
    assert policy(bq_state=bq_state_no_data, rng=rng).shape == (batch_size, 1)

    # bq state contains data
    nevals = 5
    bq_state = BQState(
        measure=measure,
        kernel=ExpQuad(input_shape=(1,)),
        nodes=np.zeros([nevals, 1]),
        fun_evals=np.ones(nevals),
    )
    assert policy(bq_state=bq_state, rng=rng).shape == (batch_size, 1)


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
