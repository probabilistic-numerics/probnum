"""
The folder is called "ode" but this
module is "ivp" because in the future,
there might be more ode-based problems,
such as bvp.
"""

import numpy as np

from probnum.diffeq.ode.ode import ODE


__all__ = ["logistic", "fitzhughnagumo", "lotkavolterra", "IVP"]


def logistic(timespan, initrv, params=(3.0, 1.0)):
    """
    Initial value problem (IVP) based on the logistic ODE.

    The logistic ODE is defined through

    .. math:: f(t, x) = a  x  \\left(1 - \\frac{x}{b}\\right)

    for some parameters :math:`(a, b)`.
    Default is :math:`(a, b)=(3.0, 1.0)`. This implementation includes
    the Jacobian :math:`J_f` of :math:`f` as well as a closed form
    solution given by

    .. math:: f(t) = \\frac{b x_0 \\exp(a t)}{b + x_0 [\\exp(at) - 1]}

    where :math:`x_0= x(t_0)` is the initial value.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
    params : (float, float), optional
        Parameters :math:`(a, b)` for the logistic IVP.
        Default is :math:`(a, b) = (3.0, 1.0)`.

    Returns
    -------
    IVP
        IVP object describing the logistic IVP with the prescribed
        configuration.
    """

    def rhs(t, x):
        return log_rhs(t, x, params)

    def jac(t, x):
        return log_jac(t, x, params)

    def hess(t, x):
        return log_hess(t, x, params)

    def sol(t):
        return log_sol(t, params, initrv.mean())

    return IVP(timespan, initrv, rhs, jac, hess, sol)


def log_rhs(t, x, params):
    """RHS for logistic model."""
    l0, l1 = params
    return l0 * x * (1.0 - x / l1)


def log_jac(t, x, params):
    """Jacobian for logistic model."""
    l0, l1 = params
    return np.array([l0 - l0 / l1 * 2 * x])


def log_hess(t, x, params):
    """Hessian for logistic model."""
    l0, l1 = params
    return np.array([[-2 * l0 / l1]])


def log_sol(t, params, x0):
    """Solution for logistic model."""
    l0, l1 = params
    nomin = l1 * x0 * np.exp(l0 * t)
    denom = l1 + x0 * (np.exp(l0 * t) - 1)
    return nomin / denom


def fitzhughnagumo(timespan, initrv, params=(0.0, 0.08, 0.07, 1.25)):
    """
    Initial value problem (IVP) based on the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo (FHN) model is defined through

    .. math:: f(t, x) =
        \\begin{pmatrix} x_1 - \\frac{1}{3}x_1^3 - x_2 + a \\\\
        \\frac{1}{d} (x_1 + b - c x_2)  \\end{pmatrix}

    for some parameters :math:`(a, b, c, d)`.
    Default is :math:`(a, b)=(0.0, 0.08, 0.07, 1.25)`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
    params : (float, float, float, float), optional
        Parameters :math:`(a, b, c, d)` for the logistic IVP.
        Default is :math:`(a, b, c, d)=(0.0, 0.08, 0.07, 1.25)`.

    Returns
    -------
    IVP
        IVP object describing the logistic IVP with the prescribed
        configuration.
    """

    def rhs(t, x):
        return fhn_rhs(t, x, params)

    def jac(t, x):
        return fhn_jac(t, x, params)

    return IVP(timespan, initrv, rhs, jac)


def fhn_rhs(t, x, params):
    """RHS for FitzHugh-Nagumo model."""
    x1, x2 = x
    a, b, c, d = params
    return np.array([x1 - x1 ** 3 / 3 - x2 + a, (x1 + b - c * x2) / d])


def fhn_jac(t, x, params):
    """Jacobian for FitzHugh-Nagumo model."""
    x1, x2 = x
    a, b, c, d = params
    return np.array([[1 - x1 ** 2, -1], [1.0 / d, -c / d]])


def lotkavolterra(timespan, initrv, params=(0.5, 0.05, 0.5, 0.05)):
    """
    Initial value problem (IVP) based on the Lotka-Volterra model.

    The Lotka-Volterra (LV) model is defined through

    .. math:: f(t, x) =
        \\begin{pmatrix} a x_1 - bx_1x_2 \\\\ -c x_2 + d x_1 x_2
        \\end{pmatrix}

    for some parameters :math:`(a, b, c, d)`.
    Default is :math:`(a, b)=(0.5, 0.05, 0.5, 0.05)`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
    params : (float, float, float, float), optional
        Parameters :math:`(a, b, c, d)` for the logistic IVP.
        Default is :math:`(a, b, c, d)=(0.5, 0.05, 0.5, 0.05)`.

    Returns
    -------
    IVP
        IVP object describing the logistic IVP with the prescribed
        configuration.
    """

    def rhs(t, x):
        return lv_rhs(t, x, params)

    def jac(t, x):
        return lv_jac(t, x, params)

    return IVP(timespan, initrv, rhs, jac)


def lv_rhs(t, x, params):
    """RHS for Lotka-Volterra"""
    a, b, c, d = params
    x1, x2 = x
    return np.array([a * x1 - b * x1 * x2, -c * x2 + d * x1 * x2])


def lv_jac(t, x, params):
    """Jacobian for Lotka-Volterra"""
    a, b, c, d = params
    x1, x2 = x
    return np.array([[a - b * x2, -b * x1], [d * x2, -c + d * x1]])


class IVP(ODE):
    """
    Initial value problems (IVP).

    This class descibes initial value problems based on systems of
    first order ordinary differential equations (ODEs),

    .. math:: \\dot x(t) = f(t, x(t)), \\quad x(t_0) = x_0,
        \\quad t \\in [t_0, T]

    It provides options for defining custom right-hand side (RHS)
    functions, their Jacobians and closed form solutions.

    Since we use them for probabilistic ODE solvers these functions
    fit into the probabilistic framework as well. That is,
    the initial value is a RandomVariable object with some
    distribution that reflects the prior belief over the initial
    value. To recover "classical" initial values one can use the
    Dirac distribution.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
        Implementation depends on the mean of this RandomVariable,
        so please only use RandomVariable objects with available
        means, e.g. Diracs or Normals.
    rhs : callable, signature: ``(t, x, **kwargs)``
        RHS function
        :math:`f : [0, T] \\times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
        of the ODE system. As such it takes a float and an
        np.ndarray of shape (d,) and returns a np.ndarray
        of shape (d,). As of now, no vectorization is supported
        (nor needed).
    jac : callable, signature: ``(t, x, **kwargs)``, optional
        Jacobian of RHS function
        :math:`J_f : [0, T] \\times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
        of the ODE system. As such it takes a float and an
        np.ndarray of shape (d,) and returns a np.ndarray
        of shape (d,). As of now, no vectorization is supported
        (nor needed).
    sol : callable, signature: ``(t, **kwargs)``, optional
        Solution of IVP.

    See Also
    --------
    ODE : Abstract interface for  ordinary differential equations.

    Examples
    --------
    >>> from probnum.diffeq import IVP
    >>> rhsfun = lambda t, x, **kwargs: 2.0*x
    >>> from probnum.prob import Dirac, Normal, RandomVariable
    >>> initrv = RandomVariable(distribution=Dirac(0.1))
    >>> timespan = (0, 10)
    >>> ivp = IVP(timespan, initrv, rhsfun)
    >>> print(ivp.rhs(0., 2.))
    4.0
    >>> print(ivp.timespan)
    [0, 10]
    >>> print(ivp.t0)
    0

    >>> initrv = RandomVariable(distribution=Normal(0.1, 1.0))
    >>> ivp = IVP(timespan, initrv, rhsfun)
    >>> jac = lambda t, x, **kwargs: 2.0
    >>> ivp = IVP(timespan, initrv, rhs=rhsfun, jac=jac)
    >>> print(ivp.rhs(0., 2.))
    4.0
    >>> print(ivp.jacobian(100., -1))
    2.0
    """

    def __init__(self, timespan, initrv, rhs, jac=None, hess=None, sol=None):
        """
        """
        self.initrv = initrv
        super().__init__(timespan=timespan, rhs=rhs, jac=jac, hess=hess, sol=sol)

    @property
    def initialdistribution(self):
        """
        Distribution of the initial random variable.
        """
        return self.initrv

    @property
    def initialrandomvariable(self):
        """
        Initial random variable.
        """
        return self.initrv

    @property
    def ndim(self):
        """
        Spatial dimension of the IVP problem.

        Depends on the mean of the initial random variable.
        """
        if np.isscalar(self.initrv.mean()):
            return 1
        else:
            return len(self.initrv.mean())
