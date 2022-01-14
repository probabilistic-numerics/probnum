"""Tests for initialization routines."""


import numpy as np
import pytest
import pytest_cases
from jax.config import config  # speed...

from probnum import randprocs
from probnum.diffeq.odefilter import init
from probnum.problems.zoo import diffeq as diffeq_zoo

from . import known_initial_derivatives

config.update("jax_disable_jit", True)


@pytest.fixture
def ivp():
    return


@pytest_cases.case(tags=("jax",))
def problem_threebody():
    ivp = diffeq_zoo.threebody_jax()
    threebody_inits_matrix_full = known_initial_derivatives.THREEBODY_INITS
    return ivp, threebody_inits_matrix_full


@pytest_cases.case(tags=("numpy",))
def problem_lotka_volterra():
    ivp = diffeq_zoo.lotkavolterra()
    inits_matrix_full = known_initial_derivatives.LV_INITS
    return ivp, inits_matrix_full


@pytest_cases.case(tags=["is_exact", "requires_jax"])
def solver_taylor_mode():
    return init.TaylorMode()


@pytest_cases.case(tags=["is_exact", "requires_jax"])
def solver_auto_diff():
    return init.AutoDiff()


@pytest_cases.case(tags=["is_not_exact", "requires_numpy"])
def solver_runge_kutta():
    return init.RungeKutta()


@pytest.mark.parametrize("num_derivatives", [0, 1, 2, 3, 5])
@pytest_cases.parametrize_with_cases(
    "ivp, dy0_true", cases=".", prefix="problem_", has_tag="jax"
)
@pytest_cases.parametrize_with_cases(
    "routine", cases=".", prefix="solver_", has_tag=("is_exact", "requires_jax")
)
def test_compare_to_reference_values_is_exact_jax(
    ivp,
    dy0_true,
    routine,
    num_derivatives,
):
    dy0_true = _select_derivatives(dy0=dy0_true, n=num_derivatives + 1)
    dy0_approximated = _compute_approximation(ivp, num_derivatives, routine)

    np.testing.assert_allclose(dy0_approximated.mean, dy0_true)
    np.testing.assert_allclose(dy0_approximated.std, 0.0)


#


@pytest.mark.parametrize("num_derivatives", [3, 4])
@pytest_cases.parametrize_with_cases(
    "ivp, dy0_true", cases=".", prefix="problem_", has_tag="numpy"
)
@pytest_cases.parametrize_with_cases(
    "routine", cases=".", prefix="solver_", has_tag=("is_not_exact", "requires_numpy")
)
def test_compare_to_reference_values_is_not_exact_numpy(
    ivp,
    dy0_true,
    routine,
    num_derivatives,
):
    dy0_true = _select_derivatives(dy0=dy0_true, n=num_derivatives + 1)
    dy0_approximated = _compute_approximation(ivp, num_derivatives, routine)

    np.testing.assert_allclose(dy0_approximated.mean, dy0_true, rtol=0.25)
    assert np.linalg.norm(dy0_approximated.std) > 0.0


def _select_derivatives(*, dy0, n):
    return dy0[:n].reshape((-1,), order="F")


def _compute_approximation(ivp, num_derivatives, routine):
    prior_process = randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0,
        num_derivatives=num_derivatives,
        wiener_process_dimension=ivp.dimension,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    dy0_approximated = routine(ivp=ivp, prior_process=prior_process)
    return dy0_approximated
