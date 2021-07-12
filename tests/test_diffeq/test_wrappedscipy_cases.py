import numpy as np
from scipy.integrate._ivp import rk

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq


def setup_solver(y0, ode):
    scipysolver = rk.RK45(ode.f, ode.t0, y0, ode.tmax)

    # Pass the identical scipysolver into the wrapper
    # (but make it a new object to avoid side effects)
    scipysolver_copy = rk.RK45(ode.f, ode.t0, y0, ode.tmax)
    testsolver = diffeq.WrappedScipyRungeKutta(scipysolver_copy)
    return testsolver, scipysolver


def case_lorenz():
    y0 = np.array([0.0, 1.0, 1.05])
    ode = diffeq_zoo.lorenz(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode)


def case_logistic():
    y0 = np.array([0.1])
    ode = diffeq_zoo.logistic(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode)


def case_lotkavolterra():
    y0 = np.array([0.1, 0.1])
    ode = diffeq_zoo.lotkavolterra(t0=0.0, tmax=1.0, y0=y0)
    return setup_solver(y0, ode)
