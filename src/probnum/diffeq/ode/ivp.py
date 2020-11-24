"""The folder is called "ode" but this module is "ivp" because in the future, there
might be more ode-based problems, such as bvp."""
# pylint: disable=unused-variable

import numpy as np

from probnum.diffeq.ode.ode import ODE

__all__ = [
    "logistic",
    "fitzhughnagumo",
    "lotkavolterra",
    "seir",
    "rigidbody",
    "vanderpol",
    "threebody",
    "IVP",
]


def logistic(timespan, initrv, params=(3.0, 1.0)):
    """Initial value problem (IVP) based on the logistic ODE.

    The logistic ODE is defined through

    .. math:: f(t, y) = a  y  \\left(1 - \\frac{y}{b}\\right)

    for some parameters :math:`(a, b)`.
    Default is :math:`(a, b)=(3.0, 1.0)`. This implementation includes
    the Jacobian :math:`J_f` of :math:`f` as well as a closed form
    solution given by

    .. math:: f(t) = \\frac{b y_0 \\exp(a t)}{b + y_0 [\\exp(at) - 1]}

    where :math:`y_0= y(t_0)` is the initial value.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Constant (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Constant distribution.
    params : (float, float), optional
        Parameters :math:`(a, b)` for the logistic IVP.
        Default is :math:`(a, b) = (3.0, 1.0)`.

    Returns
    -------
    IVP
        IVP object describing the logistic IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return log_rhs(t, y, params)

    def jac(t, y):
        return log_jac(t, y, params)

    def hess(t, y):
        return log_hess(t, y, params)

    def sol(t):
        return log_sol(t, params, initrv.mean)

    return IVP(timespan, initrv, rhs, jac, hess, sol)


def log_rhs(t, y, params):
    """RHS for logistic model."""
    l0, l1 = params
    return l0 * y * (1.0 - y / l1)


def log_jac(t, y, params):
    """Jacobian for logistic model."""
    l0, l1 = params
    return np.array([l0 - l0 / l1 * 2 * y])


def log_hess(t, y, params):
    """Hessian for logistic model."""
    l0, l1 = params
    return np.array([[-2 * l0 / l1]])


def log_sol(t, params, y0):
    """Solution for logistic model."""
    l0, l1 = params
    nomin = l1 * y0 * np.exp(l0 * t)
    denom = l1 + y0 * (np.exp(l0 * t) - 1)
    return nomin / denom


def fitzhughnagumo(timespan, initrv, params=(0.0, 0.08, 0.07, 1.25)):
    """Initial value problem (IVP) based on the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo (FHN) model is defined through

    .. math:: f(t, y) =
        \\begin{pmatrix} y_1 - \\frac{1}{3}y_1^3 - y_2 + a \\\\
        \\frac{1}{d} (y_1 + b - c y_2)  \\end{pmatrix}

    for some parameters :math:`(a, b, c, d)`.
    Default is :math:`(a, b)=(0.0, 0.08, 0.07, 1.25)`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Constant (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Constant distribution.
    params : (float, float, float, float), optional
        Parameters :math:`(a, b, c, d)` for the logistic IVP.
        Default is :math:`(a, b, c, d)=(0.0, 0.08, 0.07, 1.25)`.

    Returns
    -------
    IVP
        IVP object describing the logistic IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return fhn_rhs(t, y, params)

    def jac(t, y):
        return fhn_jac(t, y, params)

    return IVP(timespan, initrv, rhs, jac)


def fhn_rhs(t, y, params):
    """RHS for FitzHugh-Nagumo model."""
    y1, y2 = y
    a, b, c, d = params
    return np.array([y1 - y1 ** 3.0 / 3.0 - y2 + a, (y1 + b - c * y2) / d])


def fhn_jac(t, y, params):
    """Jacobian for FitzHugh-Nagumo model."""
    y1, y2 = y
    a, b, c, d = params
    return np.array([[1.0 - y1 ** 2.0, -1.0], [1.0 / d, -c / d]])


def lotkavolterra(timespan, initrv, params=(0.5, 0.05, 0.5, 0.05)):
    """Initial value problem (IVP) based on the Lotka-Volterra model.

    The Lotka-Volterra (LV) model is defined through

    .. math:: f(t, y) =
        \\begin{pmatrix} a y_1 - by_1y_2 \\\\ -c y_2 + d y_1 y_2
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
        value. Usually its distribution is Constant (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Constant distribution.
    params : (float, float, float, float), optional
        Parameters :math:`(a, b, c, d)` for the logistic IVP.
        Default is :math:`(a, b, c, d)=(0.5, 0.05, 0.5, 0.05)`.

    Returns
    -------
    IVP
        IVP object describing the logistic IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return lv_rhs(t, y, params)

    def jac(t, y):
        return lv_jac(t, y, params)

    return IVP(timespan, initrv, rhs, jac)


def lv_rhs(t, y, params):
    """RHS for Lotka-Volterra."""
    a, b, c, d = params
    y1, y2 = y
    return np.array([a * y1 - b * y1 * y2, -c * y2 + d * y1 * y2])


def lv_jac(t, y, params):
    """Jacobian for Lotka-Volterra."""
    a, b, c, d = params
    y1, y2 = y
    return np.array([[a - b * y2, -b * y1], [d * y2, -c + d * y1]])


def seir(timespan, initrv, params=(0.3, 0.3, 0.1, 1e7)):
    r"""
    Initial value problem (IVP) based on the SEIR model.

    The SEIR model with no vital dynamics is defined through

    .. math:: f(t, y) =
        \begin{pmatrix}
        \frac{-\beta y_1 y_3}{N} \\
        \frac{\beta y_1 y_3}{N} - \alpha y_2 \\
        \alpha y_2 - \gamma y_3 \\
        \gamma y_3
        \end{pmatrix}

    for some parameters :math:`(\alpha, \beta, \gamma, N)`.
    N is a constant population count.
    Default is :math:`(\alpha, \beta, \gamma, N)=(0.3, 0.3, 0.1, 10^7)`.

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
        Parameters :math:`(\alpha, \beta, \gamma, N)` for the SEIR model IVP.
        Default is :math:`(\alpha, \beta, \gamma, N)=(0.3, 0.3, 0.1, 10^7)`.

    Returns
    -------
    IVP
        IVP object describing the SEIR model IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return seir_rhs(t, y, params)

    return IVP(timespan, initrv, rhs)


def seir_rhs(t, y, params):
    alpha, beta, gamma, N = params
    y1, y2, y3, y4 = y
    y1_next = -beta * y1 * y3 / N
    y2_next = beta * y1 * y3 / N - alpha * y2
    y3_next = alpha * y2 - gamma * y3
    y4_next = gamma * y3

    return np.array([y1_next, y2_next, y3_next, y4_next])


def rigidbody(timespan, initrv):
    r"""
    Initial value problem (IVP) for rigid body dynamics without external forces

    The rigid body dynamics without external forces is defined through

    .. math:: f(t, y) =
        \begin{pmatrix}
        y_2 y_3 \\
        -y_1 y_3 \\
        -0.51 \cdot y_1 y_2
        \end{pmatrix}

    The ODE system has no parameters.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
    Returns
    -------
    IVP
        IVP object describing the rigid body dynamics IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return rigidbody_rhs(t, y)

    def jac(t, y):
        return rigidbody_jac(t, y)

    return IVP(timespan, initrv, rhs, jac=jac)


def rigidbody_rhs(t, y):
    y1, y2, y3 = y
    return np.array([y2 * y3, -y1 * y3, -0.51 * y1 * y2])


def rigidbody_jac(t, y):
    y1, y2, y3 = y
    return np.array([[0.0, y3, y2], [-y3, 0.0, -y1], [-0.51 * y2, -0.51 * y1, 0.0]])


def vanderpol(timespan, initrv, params=0.1):
    r"""
    Initial value problem (IVP) based on the Van der Pol Oscillator.

    The Van der Pol Oscillator is defined through

    .. math:: f(t, y) =
        \begin{pmatrix}
        y_2 \\
        \mu \cdot (1 - y_1^2)y_2 - y_1
        \end{pmatrix}

    for a constant parameter  :math:`\mu`.
    :math:`\mu` determines the stiffness of the problem, where
    the larger :math:`\mu` is chosen, the more stiff the problem becomes.
    Default is :math:`\mu = 0.1`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
    params : (float), optional
        Parameter :math:`\mu` for the Van der Pol Equations
        Default is :math:`\mu=0.1`.

    Returns
    -------
    IVP
        IVP object describing the Van der Pol Oscillator IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return vanderpol_rhs(t, y, params)

    def jac(t, y):
        return vanderpol_jac(t, y, params)

    return IVP(timespan, initrv, rhs, jac=jac)


def vanderpol_rhs(t, y, params):
    y1, y2 = y
    if isinstance(params, float):
        mu = params
    else:
        (mu,) = params

    return np.array([y2, mu * (1.0 - y1 ** 2) * y2 - y1])


def vanderpol_jac(t, y, params):
    y1, y2 = y
    if isinstance(params, float):
        mu = params
    else:
        (mu,) = params

    return np.array([[0.0, 1.0], [-2.0 * mu * y2 * y1 - 1.0, mu * (1.0 - y1 ** 2)]])


def threebody(timespan, initrv, params=0.012277471):
    r"""
    Initial value problem (IVP) based on a three-body problem.

    The three-body problem is defined as follows:
    Let the initial conditions be :math:`y = (y_1, y_2, \dot{y}_1, \dot{y}_2)^T`.

    .. math::

        f(t, y) =
        \begin{pmatrix}
            y_1 + 2 \dot{y}_2 - \frac{(1 - \mu) (y_1 + \mu)}{d_1} - \frac{\mu (y_1 - (1 - \mu))}{d_2} \\
            y_2 - 2 \dot{y}_1 - \frac{(1 - \mu) y_2}{d_1} - \frac{\mu y_2}{d_2}
        \end{pmatrix}

    with

    .. math::

        d_1 &= ((y_1 + \mu)^2 + y_2^2)^{\frac{3}{2}} \\
        d_2 &= ((y_1 - (1 - \mu))^2 + y_2^2)^{\frac{3}{2}}

    and a constant parameter  :math:`\mu` denoting the standardized moon mass.
    Default is :math:`\mu = 0.012277471`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Dirac (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Dirac distribution.
    params : (float), optional
        Parameter :math:`\mu` for the three-body problem
        Default is :math:`\mu = 0.012277471`.

    Returns
    -------
    IVP
        IVP object describing a three-body problem IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return threebody_rhs(t, y, params)

    return IVP(timespan, initrv, rhs)


def threebody_rhs(t, y, params):
    y1, y2, y1_dot, y2_dot = y
    if isinstance(params, float):
        mu = params
    else:
        (mu,) = params
    mp = 1.0 - mu
    d1 = ((y1 + mu) ** 2 + y2 ** 2) ** 1.5
    d2 = ((y1 - mp) ** 2 + y2 ** 2) ** 1.5

    y1p = y1 + 2.0 * y2_dot - mp * (y1 + mu) / d1 - mu * (y1 - mp) / d2
    y2p = y2 - 2.0 * y1_dot - mp * y2 / d1 - mu * y2 / d2
    return np.array([y1_dot, y2_dot, y1p, y2p])


class IVP(ODE):
    """Initial value problems (IVP).

    This class descibes initial value problems based on systems of
    first order ordinary differential equations (ODEs),

    .. math:: \\dot y(t) = f(t, y(t)), \\quad y(t_0) = y_0,
        \\quad t \\in [t_0, T]

    It provides options for defining custom right-hand side (RHS)
    functions, their Jacobians and closed form solutions.

    Since we use them for probabilistic ODE solvers these functions
    fit into the probabilistic framework as well. That is,
    the initial value is a RandomVariable object with some
    distribution that reflects the prior belief over the initial
    value. To recover "classical" initial values one can use the
    Constant distribution.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        RandomVariable that  describes the belief over the initial
        value. Usually its distribution is Constant (noise-free)
        or Normal (noisy). To replicate "classical" initial values
        use the Constant distribution.
        Implementation depends on the mean of this RandomVariable,
        so please only use RandomVariable objects with available
        means, e.g. Constants or Normals.
    rhs : callable, signature: ``(t, y, **kwargs)``
        RHS function
        :math:`f : [0, T] \\times \\mathbb{R}^d \\rightarrow \\mathbb{R}^d`
        of the ODE system. As such it takes a float and an
        np.ndarray of shape (d,) and returns a np.ndarray
        of shape (d,). As of now, no vectorization is supported
        (nor needed).
    jac : callable, signature: ``(t, y, **kwargs)``, optional
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
    >>> rhsfun = lambda t, y, **kwargs: 2.0*y
    >>> from probnum import random_variables as rvs
    >>> initrv = rvs.Constant(0.1)
    >>> timespan = (0, 10)
    >>> ivp = IVP(timespan, initrv, rhsfun)
    >>> print(ivp.rhs(0., 2.))
    4.0
    >>> print(ivp.timespan)
    [0, 10]
    >>> print(ivp.t0)
    0

    >>> initrv = rvs.Normal(0.1, 1.0)
    >>> ivp = IVP(timespan, initrv, rhsfun)
    >>> jac = lambda t, y, **kwargs: 2.0
    >>> ivp = IVP(timespan, initrv, rhs=rhsfun, jac=jac)
    >>> print(ivp.rhs(0., 2.))
    4.0
    >>> print(ivp.jacobian(100., -1))
    2.0
    """

    def __init__(self, timespan, initrv, rhs, jac=None, hess=None, sol=None):

        self.initrv = initrv
        super().__init__(timespan=timespan, rhs=rhs, jac=jac, hess=hess, sol=sol)

    @property
    def initialdistribution(self):
        """Distribution of the initial random variable."""
        return self.initrv

    @property
    def initialrandomvariable(self):
        """Initial random variable."""
        return self.initrv

    @property
    def dimension(self):
        """Spatial dimension of the IVP problem.

        Depends on the mean of the initial random variable.
        """
        if np.isscalar(self.initrv.mean):
            return 1
        else:
            return len(self.initrv.mean)
