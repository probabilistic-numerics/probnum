import numpy as np
import pytest
from scipy.integrate._ivp import rk

from probnum import diffeq


def setup_solver(y0, ode, perturbfun):
    testsolver = diffeq.WrappedScipyRungeKutta(rk.RK45(ode.rhs, ode.t0, y0, ode.tmax))
    testsolver2 = diffeq.WrappedScipyRungeKutta(rk.RK45(ode.rhs, ode.t0, y0, ode.tmax))
    perturbedsolver = diffeq.PerturbedStepSolver(
        testsolver2, noise_scale=1, perturb_function=perturbfun
    )
    return testsolver, perturbedsolver


@pytest.mark.parametrize(
    "perturbfun",
    [
        diffeq.perturb_lognormal,
        diffeq.perturb_uniform,
    ],
)
def case_lorenz(perturbfun):
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq.lorenz([0.0, 1.0], y0)
    return setup_solver(y0, ode, perturbfun)


@pytest.mark.parametrize(
    "perturbfun",
    [
        diffeq.perturb_lognormal,
        diffeq.perturb_uniform,
    ],
)
def case_logistic(perturbfun):
    y0 = np.array([0.1])
    ode = diffeq.logistic([0.0, 1.0], y0)
    return setup_solver(y0, ode, perturbfun)


@pytest.mark.parametrize(
    "perturbfun",
    [
        diffeq.perturb_lognormal,
        diffeq.perturb_uniform,
    ],
)
def case_lotkavolterra(perturbfun):
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    return setup_solver(y0, ode, perturbfun)
