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
        Parameters :math:`(a, b, c, d)` for the FitzHugh-Nagumo IVP.
        Default is :math:`(a, b, c, d)=(0.0, 0.08, 0.07, 1.25)`.

    Returns
    -------
    IVP
        IVP object describing the FitzHugh-Nagumo IVP with the prescribed
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
        Parameters :math:`(a, b, c, d)` for the Lotka-Volterra IVP.
        Default is :math:`(a, b, c, d)=(0.5, 0.05, 0.5, 0.05)`.

    Returns
    -------
    IVP
        IVP object describing the Lotka-Volterra
        IVP with the prescribed
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
    """RHS for SEIR model."""
    alpha, beta, gamma, population_count = params
    y1, y2, y3, y4 = y
    y1_next = -beta * y1 * y3 / population_count
    y2_next = beta * y1 * y3 / population_count - alpha * y2
    y3_next = alpha * y2 - gamma * y3
    y4_next = gamma * y3

    return np.array([y1_next, y2_next, y3_next, y4_next])


def seir_jac(t, y, params):
    """Jacobian for SEIR model."""
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


def lorenz(timespan, initrv, params=(10.0, 28.0, 8.0 / 3.0)):
    r"""Initial value problem (IVP) based on the Lorenz system.

    The Lorenz system is defined through

    .. math::

        f(t, y) =
        \begin{pmatrix}
            a(y_2 - y_1) \\
            y_1(b-y_3) - y_2 \\
            y_1y_2 - cy_3
        \end{pmatrix}

    for some parameters :math:`(a, b, c)`.
    Default is :math:`(a, b, c)=(10, 28, 2.667)`.
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
    params : (float, float, float, float), optional
        Parameters :math:`(a, b, c)` for the Lorenz system.
        Default is :math:`(a, b, c)=(10, 28, 2.667)`.

    Returns
    -------
    IVP
        IVP object describing the Lorenz system IVP with the prescribed
        configuration.
    """

    def rhs(t, y):
        return lor_rhs(t, y, params)

    def jac(t, y):
        return lor_jac(t, y, params)

    return IVP(timespan, initrv, rhs, jac)


def lor_rhs(t, y, params):
    """RHS for Lorenz system."""
    a, b, c = params
    y1, y2, y3 = y
    return np.array([a * (y2 - y1), y1 * (b - y3) - y2, y1 * y2 - c * y3])


def lor_jac(t, y, params):
    """Jacobian for Lorenz system."""
    a, b, c = params
    y1, y2, y3 = y
    return np.array([[-a, a, 0], [b - y3, -1, -y1], [y2, y1, -c]])
