import numpy as np
import pytest
from scipy.integrate._ivp import rk

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq

_ADAPTIVE_STEPS = diffeq.stepsize.AdaptiveSteps(atol=1e-4, rtol=1e-4, firststep=0.1)
_CONSTANT_STEPS = diffeq.stepsize.ConstantSteps(0.1)


def setup_solver(y0, ode, steprule):
    scipysolver = rk.RK45(ode.f, ode.t0, y0, ode.tmax)
    testsolver = diffeq.perturbed.scipy_wrapper.WrappedScipyRungeKutta(
        solver_type=rk.RK45, steprule=steprule
    )
    return testsolver, scipysolver, ode


@pytest.mark.parametrize("steprule", [_ADAPTIVE_STEPS, _CONSTANT_STEPS])
def case_lorenz(steprule):
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq_zoo.lorenz(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode, steprule=steprule)


@pytest.mark.parametrize("steprule", [_ADAPTIVE_STEPS, _CONSTANT_STEPS])
def case_logistic(steprule):
    y0 = np.array([0.1])
    ode = diffeq_zoo.logistic(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode, steprule=steprule)


@pytest.mark.parametrize("steprule", [_ADAPTIVE_STEPS, _CONSTANT_STEPS])
def case_lotkavolterra(steprule):
    y0 = np.array([0.1, 0.1])
    ode = diffeq_zoo.lotkavolterra(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode, steprule=steprule)
