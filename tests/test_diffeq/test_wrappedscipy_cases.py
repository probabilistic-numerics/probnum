import numpy as np
from scipy.integrate._ivp import rk

from probnum import diffeq
from probnum.diffeq import wrappedscipysolver


def setup_solver(y0, ode):
    scipysolver = rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    testsolver = wrappedscipysolver.WrappedScipyRungeKutta(
        rk.RK45(ode.rhs, ode.t0, y0, ode.tmax)
    )
    return testsolver, scipysolver


def case_lorenz():
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq.lorenz([0.0, 1.0], y0)
    return setup_solver(y0, ode)


def case_logistic():
    y0 = np.array([0.1])
    ode = diffeq.logistic([0.0, 1.0], y0)
    return setup_solver(y0, ode)


def case_lotkavolterra():
    y0 = np.array([0.1, 0.1])
    ode = diffeq.lotkavolterra([0.0, 1.0], y0)
    return setup_solver(y0, ode)
