"""Tests for ODE filters."""


import pytest
import pytest_cases

from probnum import diffeq, randprocs


@pytest.mark.parametrize(
    "num_derivatives",
    [
        3,
    ],
)
@pytest.mark.parametrize("with_smoothing", [True, False])
@pytest_cases.parametrize_with_cases("steprule", prefix="steprule_")
@pytest_cases.parametrize_with_cases("diffusion_model", prefix="diffusion_")
@pytest_cases.parametrize_with_cases("approx_strategy", prefix="approx_")
@pytest_cases.parametrize_with_cases("ivp", prefix="problem_", has_tag="numpy")
@pytest_cases.parametrize_with_cases("init", prefix="init_", has_tag="numpy")
def test_solve_numpy(
    ivp,
    steprule,
    num_derivatives,
    with_smoothing,
    diffusion_model,
    init,
    approx_strategy,
):

    solver = diffeq.odefilter.ODEFilter(
        steprule=steprule,
        prior_process=_prior_process(ivp=ivp, num_derivatives=num_derivatives),
        with_smoothing=with_smoothing,
        diffusion_model=diffusion_model,
        init_routine=init,
        approx_strategy=approx_strategy,
    )

    solver.solve(ivp)


@pytest.mark.parametrize(
    "num_derivatives",
    [
        3,
    ],
)
@pytest.mark.parametrize("with_smoothing", [True, False])
@pytest_cases.parametrize_with_cases("steprule", prefix="steprule_")
@pytest_cases.parametrize_with_cases("diffusion_model", prefix="diffusion_")
@pytest_cases.parametrize_with_cases("approx_strategy", prefix="approx_")
@pytest_cases.parametrize_with_cases("ivp", prefix="problem_", has_tag="jax")
@pytest_cases.parametrize_with_cases("init", prefix="init_", has_tag="jax")
def test_solve_jax(
    ivp,
    steprule,
    num_derivatives,
    with_smoothing,
    diffusion_model,
    init,
    approx_strategy,
):

    solver = diffeq.odefilter.ODEFilter(
        steprule=steprule,
        prior_process=_prior_process(ivp=ivp, num_derivatives=num_derivatives),
        with_smoothing=with_smoothing,
        diffusion_model=diffusion_model,
        init_routine=init,
        approx_strategy=approx_strategy,
    )
    solver.solve(ivp)


def _prior_process(*, ivp, num_derivatives):
    return randprocs.markov.integrator.IntegratedWienerProcess(
        initarg=ivp.t0,
        num_derivatives=num_derivatives,
        wiener_process_dimension=ivp.dimension,
        diffuse=True,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
