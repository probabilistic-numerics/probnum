"""Tests for Markov processes."""

import numpy as np

from probnum import backend, randprocs, randvars

import pytest


def test_bad_args_shape():
    time_domain = (0.0, 10.0)
    time_grid = np.arange(*time_domain)

    order = 2
    spatialdim = 2

    dynamics = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=order,
        wiener_process_dimension=spatialdim,
    )
    initrv = randvars.Normal(
        np.ones(dynamics.state_dimension),
        np.eye(dynamics.state_dimension),
    )
    prior_process = randprocs.markov.MarkovProcess(
        initarg=time_domain[0], initrv=initrv, transition=dynamics
    )

    with pytest.raises(ValueError):
        prior_process.sample(
            rng_state=backend.random.rng_state(1), args=time_grid.reshape(-1, 1)
        )
