import numpy as np
import pytest
from pn_ode_benchmarks import wrapperscipysolver as wrappedscipysolver
from scipy.integrate._ivp import rk

from probnum import diffeq
from probnum.diffeq.perturbedsolvers import _perturbation_functions, perturbedstepsolver


def setup_solver(y0, ode, steprule):
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    )
    testsolver2 = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    )
    perturbfun = steprule
    perturbedsolver = perturbedstepsolver.PerturbedStepSolver(
        testsolver2, noise_scale=10, perturb_function=perturbfun
    )
    return testsolver, perturbedsolver


@pytest.mark.parametrize(
    "steprule",
    [
        _perturbation_functions.perturb_lognormal,
        _perturbation_functions.perturb_uniform,
    ],
)
def case_lorenz(steprule):
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq.lorenz([0.0, 1.0], y0)
    return setup_solver(y0, ode, steprule)


@pytest.mark.parametrize(
    "steprule",
    [
        _perturbation_functions.perturb_lognormal,
        _perturbation_functions.perturb_uniform,
    ],
)
def case_logistic(steprule):
    y0 = np.array([0.1])
    ode = diffeq.logistic([0.0, 1.0], y0)
    return setup_solver(y0, ode, steprule)


@pytest.mark.parametrize(
    "steprule",
    [
        _perturbation_functions.perturb_lognormal,
        _perturbation_functions.perturb_uniform,
    ],
)
def case_lotkavolterra(steprule):
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    return setup_solver(y0, ode, steprule)
