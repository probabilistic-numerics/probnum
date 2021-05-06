"""Tests for Markov processes."""

from probnum import randprocs, randvars


def test_gauss_markov_process_initrv_is_gaussian(
    gauss_markov_process: randprocs.MarkovProcess,
):
    """Test whether Gauss-Markov processes are initialized with a Gaussian random
    variable."""
    assert isinstance(
        gauss_markov_process.initrv,
        randvars.Normal,
    )
