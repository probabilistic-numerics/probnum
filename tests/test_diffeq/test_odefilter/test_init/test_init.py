"""Tests for initialization routines."""


import jax
import numpy as np
import pytest
import pytest_cases
from jax.config import config  # speed...

from probnum import randprocs

config.update("jax_disable_jit", True)


@pytest.mark.parametrize("num_derivatives", [2, 3, 5])
@pytest_cases.parametrize_with_cases("ivp, dy0_true", prefix="problem_", has_tag="jax")
@pytest_cases.parametrize_with_cases(
    "routine", prefix="solver_", has_tag=("is_exact", "requires_jax")
)
def test_compare_to_reference_values_is_exact_jax(
    ivp,
    dy0_true,
    routine,
    num_derivatives,
):
    dy0_true = _select_derivatives(dy0=dy0_true, n=num_derivatives + 1)

    dy0_approximated, _ = _compute_approximation(ivp, num_derivatives, routine)

    np.testing.assert_allclose(dy0_approximated.mean, dy0_true)
    np.testing.assert_allclose(dy0_approximated.std, 0.0)


@pytest.mark.parametrize("num_derivatives", [3, 4])
@pytest_cases.parametrize_with_cases(
    "ivp, dy0_true", prefix="problem_", has_tag="numpy"
)
@pytest_cases.parametrize_with_cases(
    "routine", prefix="solver_", has_tag=("is_not_exact", "requires_numpy")
)
def test_compare_to_reference_values_is_not_exact_numpy(
    ivp,
    dy0_true,
    routine,
    num_derivatives,
):
    dy0_true = _select_derivatives(dy0=dy0_true, n=num_derivatives + 1)
    dy0_approximated, prior_process = _compute_approximation(
        ivp, num_derivatives, routine
    )

    # The zeroth and first derivative must always be exact
    P0 = prior_process.transition.proj2coord(0)
    P1 = prior_process.transition.proj2coord(1)
    np.testing.assert_allclose(P0 @ dy0_approximated.mean, P0 @ dy0_true, rtol=1e-5)
    np.testing.assert_allclose(P1 @ dy0_approximated.mean, P1 @ dy0_true, rtol=1e-5)

    # The error in the remainder must be encoded by a positive standard deviation
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
    return dy0_approximated, prior_process
