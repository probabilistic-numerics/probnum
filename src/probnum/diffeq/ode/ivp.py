"""
The folder is called "ode" but this
module is "ivp" because in the future,
there might be more ode-based problems,
such as bvp.
"""

import numpy as np

from probnum.diffeq.ode.ode import ODE


__all__ = ["logistic", "fitzhughnagumo", "lotkavolterra", "IVP"]


def logistic(timespan, initdist, params=(3.0, 1.0)):
    """
    Logistic ODE (see Kersting et al, 2019).
    """
    def rhs(t, x):
        return log_rhs(t, x, params)

    def jac(t, x):
        return log_jac(t, x, params)

    def sol(t):
        return log_sol(t, params, initdist.mean())

    return IVP(timespan, initdist, rhs, jac, sol)


def log_rhs(t, x, params):
    """RHS for logistic model."""
    l0, l1 = params
    return l0 * x * (1.0 - x / l1)


def log_jac(t, x, params):
    """Jacobian for logistic model."""
    l0, l1 = params
    return np.array([l0 - l0/l1*2*x])


def log_sol(t, params, x0):
    """Solution for logistic model."""
    l0, l1 = params
    nomin = l1 * x0 * np.exp(l0*t)
    denom = l1 + x0 * (np.exp(l0*t) - 1)
    return nomin / denom


def fitzhughnagumo(timespan, initdist, params=(0.0, 0.08, 0.07, 1.25)):
    """
    Computes f(t, x) = (x1 - x1**3/3 - x2 + c_0,
                        1/c_3 * (x1 + c_1 - c_2 * x2))
    for parameters c = (c_0, ..., c_3).
    """

    def rhs(t, x):
        return fhn_rhs(t, x, params)

    def jac(t, x):
        return fhn_jac(t, x, params)

    return IVP(timespan, initdist, rhs, jac)


def fhn_rhs(t, x, params):
    """RHS for FitzHugh-Nagumo model."""
    x1, x2 = x
    a, b, c, d = params
    return np.array([x1 - x1 ** 3 / 3 - x2 + a,
                     (x1 - b - c * x2) / d])

def fhn_jac(t, x, params):
    """Jacobian for FitzHugh-Nagumo model."""
    x1, x2 = x
    a, b, c, d = params
    return np.array([[1 - x1**2, -1], [1.0 / d, - c / d]])


def lotkavolterra(timespan, initdist, params=(0.5, 0.05, 0.5, 0.05)):
    """
    Returns Lotka Volterra ODE.
    """

    def rhs(t, x):
        return lv_rhs(t, x, params)

    def jac(t, x):
        return lv_jac(t, x, params)

    return IVP(timespan, initdist, rhs, jac)


def lv_rhs(t, x, params):
    """RHS for Lotka-Volterra"""
    a, b, c, d = params
    x1, x2 = x
    return np.array([a * x1 - b * x1 * x2,
                     -c * x2 + d * x1 * x2])


def lv_jac(t, x, params):
    """Jacobian for Lotka-Volterra"""
    a, b, c, d = params
    x1, x2 = x
    return np.array([[a - b * x2, -b * x1], [d * x2, -c + d * x1]])


class IVP(ODE):
    """
    Basic IVP class.
    """

    def __init__(self, timespan, initdist, rhs, jac=None, sol=None):
        """
        Initialises basic ODE attributes.

        We take timespan as a list to mimic the
        scipy.solve_ivp interface.

        timespan : (t0, tmax); tuple of two floats,
            time span of IVP
        initdist : prob.RandomVariable,
            usually dirac.Dirac (noise-free)
            or gaussian.MultivariateGaussian (noisy)
        rhs : callable, signature (t, x, *args)
            right hand side vector field function
        jac : callable, signature (t, x, *args)
            Jacobian of right hand side function
        """
        self.initdist = initdist
        super().__init__(timespan, rhs, jac, sol)

    @property
    def initialdistribution(self):
        """
        """
        return self.initdist

    @property
    def ndim(self):
        """
        """
        if np.isscalar(self.initdist.mean()):
            return 1
        else:
            return len(self.initdist.mean())
