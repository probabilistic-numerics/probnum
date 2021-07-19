"""Tests for Markov processes."""
import time

import numpy as np
import pytest

from probnum import config, linops, randprocs, randvars, statespace


def test_gauss_markov_process_initrv_is_gaussian(
    gauss_markov_process: randprocs.MarkovProcess,
):
    """Test whether Gauss-Markov processes are initialized with a Gaussian random
    variable."""
    assert isinstance(
        gauss_markov_process.initrv,
        randvars.Normal,
    )


def test_sample():
    rng = np.random.default_rng(seed=1)
    time_domain = (0.0, 10.0)
    time_grid = np.arange(*time_domain)
    measvar = 0.1024

    order = 5
    spatialdim = 100

    with config(prefer_dense_arrays=True):
        dynamics_dense = statespace.IBM(
            ordint=order,
            spatialdim=spatialdim,
            forward_implementation="classic",
            backward_implementation="classic",
        )
        initrv_dense = randvars.Normal(
            np.ones(dynamics_dense.dimension),
            measvar * np.eye(dynamics_dense.dimension),
        )
        prior_process_dense = randprocs.MarkovProcess(
            initarg=time_domain[0], initrv=initrv_dense, transition=dynamics_dense
        )

        start_dense = time.time()
        prior_process_dense.sample(rng=rng, args=time_grid)
        stop_dense = time.time()

    time_dense = stop_dense - start_dense

    with config(prefer_dense_arrays=False):
        dynamics_linop = statespace.IBM(
            ordint=order,
            spatialdim=spatialdim,
            forward_implementation="classic",
            backward_implementation="classic",
        )
        initrv_linop = randvars.Normal(
            np.ones(dynamics_linop.dimension),
            measvar * linops.Identity(dynamics_linop.dimension),
            cov_cholesky=np.sqrt(measvar) * linops.Identity(dynamics_linop.dimension),
        )
        prior_process_linop = randprocs.MarkovProcess(
            initarg=time_domain[0], initrv=initrv_linop, transition=dynamics_linop
        )

        start_linop = time.time()
        prior_process_linop.sample(rng=rng, args=time_grid)
        stop_linop = time.time()

    time_linop = stop_linop - start_linop

    print(f"Dense: {time_dense}")
    print(f"LinOp: {time_linop}")

    assert time_linop < time_dense


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
