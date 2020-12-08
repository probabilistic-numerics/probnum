import numpy as np

from probnum.diffeq.ode.ivp import IVP


def logistic(timespan, initrv, params=(3.0, 1.0)):
    r"""Initial value problem (IVP) based on the logistic ODE.

    The logistic ODE is defined through

    .. math::

        f(t, y) = a  y  \left( 1 - \frac{y}{b} \right)

    for some parameters :math:`(a, b)`.
    Default is :math:`(a, b)=(3.0, 1.0)`. This implementation includes
    the Jacobian :math:`J_f` of :math:`f` as well as a closed form
    solution given by

    .. math::

        f(t) = \frac{b y_0 \exp(a t)}{b + y_0 \left[ \exp(at) - 1 \right]}

    where :math:`y_0= y(t_0)` is the initial value.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        *(shape=())* -- Scalar-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free      or Normal (no
        Random Variable isy) with scalar mean and scalar variance.
        To replicate "classical" initial values use the Constant distribution.
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
    r"""Initial value problem (IVP) based on the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo (FHN) model is defined through

    .. math::

        f(t, y) =
        \begin{pmatrix}
            y_1 - \frac{1}{3} y_1^3 - y_2 + a \\
            \frac{1}{d} (y_1 + b - c y_2)
        \end{pmatrix}

    for some parameters :math:`(a, b, c, d)`.
    Default is :math:`(a, b)=(0.0, 0.08, 0.07, 1.25)`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        *(shape=(2, ))* -- Vector-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free) or Normal (noisy)
        Random Variable with :math:`2`-dimensional mean vector and
        :math:`2 \times 2`-dimensional covariance matrix.
        To replicate "classical" initial values use the Constant distribution.
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
    r"""Initial value problem (IVP) based on the Lotka-Volterra model.

    The Lotka-Volterra (LV) model is defined through

    .. math::

        f(t, y) =
        \begin{pmatrix}
            a y_1 - by_1y_2 \\
            -c y_2 + d y_1 y_2
        \end{pmatrix}

    for some parameters :math:`(a, b, c, d)`.
    Default is :math:`(a, b)=(0.5, 0.05, 0.5, 0.05)`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        *(shape=(2, ))* -- Vector-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free) or Normal (noisy)
        Random Variable with :math:`2`-dimensional mean vector and
        :math:`2 \times 2`-dimensional covariance matrix.
        To replicate "classical" initial values use the Constant distribution.
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


def seir(timespan, initrv, params=(0.3, 0.3, 0.1)):
    r"""Initial value problem (IVP) based on the SEIR model.

    The SEIR model with no vital dynamics is defined through

    .. math::

        f(t, y) =
        \begin{pmatrix}
            \frac{-\beta y_1 y_3}{N} \\
            \frac{\beta y_1 y_3}{N} - \alpha y_2 \\
            \alpha y_2 - \gamma y_3 \\
            \gamma y_3
        \end{pmatrix}

    for some parameters :math:`(\alpha, \beta, \gamma)` and population
    count :math:`N`. Without taking vital dynamics into consideration,
    :math:`N` is constant such that for every time point :math:`t`

    .. math::

        S(t) + E(t) + I(t) + R(t) = N

    holds.
    Default parameters are :math:`(\alpha, \beta, \gamma)=(0.3, 0.3, 0.1)`.
    The population count is computed from the (mean of the)
    initial value random variable.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        *(shape=(4, ))* -- Vector-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free) or Normal (noisy)
        Random Variable with :math:`4`-dimensional mean vector and
        :math:`4 \times 4`-dimensional covariance matrix.
        To replicate "classical" initial values use the Constant distribution.
    params : (float, float, float), optional
        Parameters :math:`(\alpha, \beta, \gamma)` for the SEIR model IVP.
        Default is :math:`(\alpha, \beta, \gamma)=(0.3, 0.3, 0.1)`.

    Returns
    -------
    IVP
        IVP object describing the SEIR model IVP with the prescribed
        configuration.
    """

    population_count = np.sum(initrv.mean)
    params_and_population_count = (*params, population_count)

    def rhs(t, y):
        return seir_rhs(t, y, params_and_population_count)

    def jac(t, y):
        return seir_jac(t, y, params_and_population_count)

    return IVP(timespan, initrv, rhs, jac=jac)


def seir_rhs(t, y, params):
    """RHS for SEIR model"""
    alpha, beta, gamma, population_count = params
    y1, y2, y3, y4 = y
    y1_next = -beta * y1 * y3 / population_count
    y2_next = beta * y1 * y3 / population_count - alpha * y2
    y3_next = alpha * y2 - gamma * y3
    y4_next = gamma * y3

    return np.array([y1_next, y2_next, y3_next, y4_next])


def seir_jac(t, y, params):
    """Jacobian for SEIR model"""
    alpha, beta, gamma, population_count = params
    y1, y2, y3, y4 = y
    d_dy1 = np.array(
        [-beta * y3 / population_count, 0.0, -beta * y1 / population_count, 0.0]
    )
    d_dy2 = np.array(
        [beta * y3 / population_count, -alpha, beta * y1 / population_count, 0.0]
    )
    d_dy3 = np.array([0.0, alpha, -gamma, 0.0])
    d_dy4 = np.array([0.0, 0.0, gamma, 0.0])
    jac_matrix = np.array([d_dy1, d_dy2, d_dy3, d_dy4])
    return jac_matrix


def rigidbody(timespan, initrv):
    r"""Initial value problem (IVP) for rigid body dynamics without external forces

    The rigid body dynamics without external forces is defined through

    .. math::

        f(t, y) =
        \begin{pmatrix}
            y_2 y_3 \\
            -y_1 y_3 \\
            -0.51 \cdot y_1 y_2
        \end{pmatrix}

    The ODE system has no parameters.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        *(shape=(3, ))* -- Vector-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free) or Normal (noisy)
        Random Variable with :math:`3`-dimensional mean vector and
        :math:`3 \times 3`-dimensional covariance matrix.
        To replicate "classical" initial values use the Constant distribution.
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
    r"""Initial value problem (IVP) based on the Van der Pol Oscillator.

    This function implements the second-order Van-der-Pol Oscillator as a system
    of first-order ODEs.
    The Van der Pol Oscillator is defined as

    .. math::

        f(t, y) =
        \begin{pmatrix}
            y_2 \\
            \mu \cdot (1 - y_1^2)y_2 - y_1
        \end{pmatrix}

    for a constant parameter  :math:`\mu`.
    :math:`\mu` determines the stiffness of the problem, where
    the larger :math:`\mu` is chosen, the more stiff the problem becomes.
    Default is :math:`\mu = 0.1`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    timespan : (float, float)
        Time span of IVP.
    initrv : RandomVariable,
        *(shape=(2, ))* -- Vector-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free) or Normal (noisy)
        Random Variable with :math:`2`-dimensional mean vector and
        :math:`2 \times 2`-dimensional covariance matrix.
        To replicate "classical" initial values use the Constant distribution.
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
    r"""Initial value problem (IVP) based on a three-body problem.

    Let the initial conditions be :math:`y = (y_1, y_2, \dot{y}_1, \dot{y}_2)^T`.
    This function implements the second-order three-body problem as a system of
    first-order ODEs, which is defined as follows: [1]_

    .. math::

        f(t, y) =
        \begin{pmatrix}
            \dot{y_1} \\
            \dot{y_2} \\
            y_1 + 2 \dot{y}_2 - \frac{(1 - \mu) (y_1 + \mu)}{d_1}
                - \frac{\mu (y_1 - (1 - \mu))}{d_2} \\
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
        *(shape=(4, ))* -- Vector-valued RandomVariable that describes the belief
        over the initial value. Usually it is a Constant (noise-free) or Normal (noisy)
        Random Variable with :math:`4`-dimensional mean vector and
        :math:`4 \times 4`-dimensional covariance matrix.
        To replicate "classical" initial values use the Constant distribution.
    params : (float), optional
        Parameter :math:`\mu` for the three-body problem
        Default is :math:`\mu = 0.012277471`.

    Returns
    -------
    IVP
        IVP object describing a three-body problem IVP with the prescribed
        configuration.

    References
    ----------
    .. [1] Hairer, E., Norsett, S. and Wanner, G..
        Solving Ordinary Differential Equations I.
        Springer Series in Computational Mathematics, 1993.
    """

    def rhs(t, y):
        return threebody_rhs(t, y, params)

    return IVP(timespan, initrv, rhs)


def threebody_rhs(t, y, params):
    y1, y2, y1_dot, y2_dot = y
    if isinstance(params, float):
        standardized_moon_mass = params
    else:
        (standardized_moon_mass,) = params
    mu = standardized_moon_mass
    mp = 1.0 - mu
    d1 = ((y1 + mu) ** 2 + y2 ** 2) ** 1.5
    d2 = ((y1 - mp) ** 2 + y2 ** 2) ** 1.5

    y1p = y1 + 2.0 * y2_dot - mp * (y1 + mu) / d1 - mu * (y1 - mp) / d2
    y2p = y2 - 2.0 * y1_dot - mp * y2 / d1 - mu * y2 / d2
    return np.array([y1_dot, y2_dot, y1p, y2p])
