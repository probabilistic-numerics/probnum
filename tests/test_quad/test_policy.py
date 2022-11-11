"""Basic tests for BQ policies."""

import numpy as np
import pytest

from probnum.quad import GaussianMeasure, LebesgueMeasure, VanDerCorputPolicy, bayesquad


def test_van_der_corput_multi_d_error(input_dim):
    """Check that van der Corput policy fails in dimensions higher than one."""

    def fun(x):
        return np.ones(x.shape)

    measure = GaussianMeasure(input_dim=input_dim, mean=0.0, cov=1.0)
    if input_dim > 1:
        with pytest.raises(ValueError):
            bayesquad(fun=fun, input_dim=input_dim, measure=measure, policy="vdc")


@pytest.mark.parametrize(
    "domain", [(0, 1), (-np.Inf, 0), (1, np.Inf), (-np.Inf, np.Inf)]
)
def test_van_der_corput_infinite_error(domain):
    """Check that van der Corput policy fails on infinite domains."""

    def fun(x):
        return np.ones(x.shape)

    measure = LebesgueMeasure(input_dim=1, domain=domain)
    if domain[1] - domain[0] == np.Inf:
        with pytest.raises(ValueError):
            bayesquad(fun=fun, input_dim=1, measure=measure, policy="vdc")


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
    expected_seq = np.array([0.125 , 0.625 , 0.375 , 0.875 , 0.0625, 0.5625, 0.3125])
    np.testing.assert_array_equal(vdc_seq, expected_seq)

