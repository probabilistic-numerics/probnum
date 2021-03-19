"""Tests for Gauss-Markov processes."""

from probnum import randprocs


def test_gauss_markov_processes_are_gps(
    gauss_markov_process: randprocs.GaussMarkovProcess,
):
    """Test whether Gauss-Markov processes are Gaussian processes."""
    assert isinstance(gauss_markov_process, randprocs.GaussianProcess)
