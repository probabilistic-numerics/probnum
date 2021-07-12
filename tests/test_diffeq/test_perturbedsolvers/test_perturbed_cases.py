import numpy as np
import pytest
from scipy.integrate._ivp import rk

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq


def setup_solver(y0, ode, perturbfun):
    rng = np.random.default_rng(seed=1)
    testsolver = diffeq.WrappedScipyRungeKutta(rk.RK45(ode.f, ode.t0, y0, ode.tmax))
    testsolver2 = diffeq.WrappedScipyRungeKutta(rk.RK45(ode.f, ode.t0, y0, ode.tmax))
    perturbedsolver = diffeq.PerturbedStepSolver(
        rng=rng, solver=testsolver2, noise_scale=1.0, perturb_function=perturbfun
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
    ode = diffeq_zoo.lorenz(t0=0.0, tmax=1.0, y0=y0)
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
    ode = diffeq_zoo.logistic(t0=0.0, tmax=1.0, y0=y0)
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
    ode = diffeq_zoo.lotkavolterra(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode, perturbfun)
