"""Tests for Markov processes."""

import numpy as np
import pytest

from probnum import randprocs, randvars, statespace


def test_gauss_markov_process_initrv_is_gaussian(
    gauss_markov_process: randprocs.MarkovProcess,
):
    """Test whether Gauss-Markov processes are initialized with a Gaussian random
    variable."""
    assert isinstance(
        gauss_markov_process.initrv,
        randvars.Normal,
    )


def test_bad_args_shape():
    rng = np.random.default_rng(seed=1)
    time_domain = (0.0, 10.0)
    time_grid = np.arange(*time_domain)

    order = 2
    spatialdim = 2

    dynamics = statespace.IBM(
        ordint=order,
        spatialdim=spatialdim,
    )
    initrv = randvars.Normal(
        np.ones(dynamics.dimension),
        np.eye(dynamics.dimension),
    )
    prior_process = randprocs.MarkovProcess(
        initarg=time_domain[0], initrv=initrv, transition=dynamics
    )

    with pytest.raises(ValueError):
        prior_process.sample(rng=rng, args=time_grid.reshape(-1, 1))
