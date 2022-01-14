"""Tests for initialization routines."""


import numpy as np
import pytest
import pytest_cases

from probnum import randprocs
from probnum.diffeq.odefilter import init
from probnum.problems.zoo import diffeq as diffeq_zoo

from . import known_initial_derivatives


@pytest.fixture
def ivp():
    return diffeq_zoo.threebody_jax()


@pytest.fixture
def num_derivatives():
    return 2


@pytest.fixture
def dy0_true(num_derivatives):
    return known_initial_derivatives.THREEBODY_INITS[: num_derivatives + 1].reshape(
        (-1,), order="F"
    )


@pytest.fixture
def prior_process(ivp, num_derivatives):
    return randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0,
        num_derivatives=num_derivatives,
        wiener_process_dimension=ivp.dimension,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )


@pytest_cases.case(tags=["is_exact"])
def case_taylor_mode(
    prior_process,
    ivp,
):
    init_tm = init.TaylorMode()
    return init_tm(ivp=ivp, prior_process=prior_process)


@pytest_cases.parametrize_with_cases("dy0_approximated", cases=".", has_tag="is_exact")
def test_is_exact(
    dy0_approximated,
    dy0_true,
):
    np.testing.assert_allclose(dy0_approximated.mean, dy0_true)
    np.testing.assert_allclose(dy0_approximated.std, 0.0)
