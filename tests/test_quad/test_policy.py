"""Basic tests for BQ policies."""

import numpy as np
import pytest

from probnum.quad.integration_measures import GaussianMeasure, LebesgueMeasure
from probnum.quad.solvers.policies import VanDerCorputPolicy

# Todo: tests for other policies


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
