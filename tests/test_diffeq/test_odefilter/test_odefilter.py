"""Tests for ODE filters."""


import pytest
import pytest_cases

from probnum import diffeq, randprocs


@pytest.mark.parametrize("num_derivatives", [2, 3, 5])
@pytest.mark.parametrize("with_smoothing", [True, False])
@pytest_cases.parametrize_with_cases("ivp", prefix="problem_")
@pytest_cases.parametrize_with_cases("steprule", prefix="steprule_")
@pytest_cases.parametrize_with_cases("diffusion_model", prefix="diffusion_")
@pytest_cases.parametrize_with_cases("init", prefix="init_")
def test_solve(ivp, steprule, num_derivatives, with_smoothing, diffusion_model, init):

    solver = diffeq.odefilter.ODEFilter(
        steprule=steprule,
        prior_process=_prior_process(ivp=ivp, num_derivatives=num_derivatives),
        with_smoothing=with_smoothing,
        diffusion_model=diffusion_model,
        initialization_routine=init,
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
