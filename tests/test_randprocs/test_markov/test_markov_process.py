"""Tests for Markov processes."""

import numpy as np
import pytest

from probnum import randprocs, randvars


def test_bad_args_shape():
    rng = np.random.default_rng(seed=1)
    time_domain = (0.0, 10.0)
    time_grid = np.arange(*time_domain)

    order = 2
    spatialdim = 2

    dynamics = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=order,
        wiener_process_dimension=spatialdim,
    )
    initrv = randvars.Normal(
        np.ones(dynamics.dimension),
        np.eye(dynamics.dimension),
    )
    prior_process = randprocs.markov.MarkovProcess(
        initarg=time_domain[0], initrv=initrv, transition=dynamics
    )

    with pytest.raises(ValueError):
        prior_process.sample(rng=rng, args=time_grid.reshape(-1, 1))
