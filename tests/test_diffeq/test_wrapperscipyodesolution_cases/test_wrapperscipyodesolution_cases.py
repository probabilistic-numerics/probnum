import numpy as np
import pytest_cases
from scipy.integrate._ivp import rk

from probnum import diffeq
from probnum.diffeq import wrapperscipysolver


def case_lorenz():
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq.lorenz([0.0, 1.0], y0)
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrapperscipysolver.WrapperScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax), order=4
    )
    testsolution = testsolver.solve(diffeq.ConstantSteps(0.1))
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution


def case_logistic():
    y0 = np.array([0.1])
    ode = diffeq.logistic([0.0, 1.0], y0)
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrapperscipysolver.WrapperScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax), order=4
    )
    testsolution = testsolver.solve(diffeq.AdaptiveSteps(0.1, atol=1e-2, rtol=1e-2))
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution


def case_lotkavolterra():
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    testsolver = wrapperscipysolver.WrapperScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax), order=4
    )
    testsolution = testsolver.solve(diffeq.AdaptiveSteps(0.1, atol=1e-12, rtol=1e-12))
    scipysolution = testsolution.scipy_solution
    return testsolution, scipysolution
