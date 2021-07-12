import numpy as np

from probnum.problems import InitialValueProblem

__all__ = [
    "threebody",
    "vanderpol",
    "rigidbody",
    "fitzhughnagumo",
    "logistic",
    "lotkavolterra",
    "seir",
    "lorenz",
]


def threebody(t0=0.0, tmax=17.0652165601579625588917206249, y0=None):
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

    and a constant parameter  :math:`\mu = 0.012277471` denoting the standardized moon mass.

    Parameters
    ----------
    t0
        Initial time. Default is ``0.0``.
    tmax
        Final time. Default is ``17.0652165601579625588917206249`` which is the period of the solution.
    y0
        *(shape=(4, ))* -- Initial value. Default is ``[0.994, 0, 0, -2.00158510637908252240537862224]``.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing a three-body problem IVP with the prescribed
        configuration.

    References
    ----------
    .. [1] Hairer, E., Norsett, S. and Wanner, G..
        Solving Ordinary Differential Equations I.
        Springer Series in Computational Mathematics, 1993.
    """

    def rhs(t, y):
        mu = 0.012277471  # a constant (standardised moon mass)
        mp = 1 - mu
        D1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** (3 / 2)
        D2 = ((y[0] - mp) ** 2 + y[1] ** 2) ** (3 / 2)
        y1p = y[0] + 2 * y[3] - mp * (y[0] + mu) / D1 - mu * (y[0] - mp) / D2
        y2p = y[1] - 2 * y[2] - mp * y[1] / D1 - mu * y[1] / D2
        return np.array([y[2], y[3], y1p, y2p])

    if y0 is None:
        y0 = np.array([0.994, 0, 0, -2.00158510637908252240537862224])

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0)


def vanderpol(t0=0.0, tmax=30, y0=None, params=1e1):
    r"""Initial value problem (IVP) based on the Van der Pol Oscillator, implemented in `jax`.

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
    t0 : float
        Initial time point. Leftmost point of the integration domain.
    tmax : float
        Final time point. Rightmost point of the integration domain.
    y0 : np.ndarray,
        *(shape=(2, ))* -- Initial value of the problem.
    params : (float), optional
        Parameter :math:`\mu` for the Van der Pol Equations
        Default is :math:`\mu=0.1`.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the Van der Pol Oscillator IVP with the prescribed
        configuration.
    """

    if y0 is None:
        y0 = np.array([2.0, 0.0])

    def rhs(t, y, params=params):
        y1, y2 = y
        if isinstance(params, float):
            mu = params
        else:
            (mu,) = params

        return np.array([y2, mu * (1.0 - y1 ** 2) * y2 - y1])

    def jac(t, y, params=params):
        y1, y2 = y
        if isinstance(params, float):
            mu = params
        else:
            (mu,) = params

        return np.array([[0.0, 1.0], [-2.0 * mu * y2 * y1 - 1.0, mu * (1.0 - y1 ** 2)]])

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def rigidbody(t0=0.0, tmax=20.0, y0=None, params=(-2.0, 1.25, -0.5)):
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
    t0
        Initial time. Default is 0.0
    tmax
        Final time. Default is 20.0
    y0
        *(shape=(3, ))* -- Initial value. Default is ``[1., 0., 0.9]``.
    params
        Parameter of the rigid body problem. Default is ``(-2.0, 1.25, -0.5)``.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the rigid body dynamics IVP with the prescribed
        configuration.
    """
    if y0 is None:
        y0 = np.array([1.0, 0.0, 0.9])

    def rhs(t, y, params=params):
        p1, p2, p3 = params
        y1, y2, y3 = y
        return np.array([p1 * y2 * y3, p2 * y1 * y3, p3 * y1 * y2])

    def jac(t, y, params=params):
        p1, p2, p3 = params

        y1, y2, y3 = y
        return np.array(
            [[0.0, p1 * y3, p1 * y2], [p2 * y3, 0.0, p2 * y1], [p3 * y2, p3 * y1, 0.0]]
        )

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def logistic(t0=0.0, tmax=2.0, y0=None, params=(3.0, 1.0)):
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
    t0
        Initial time. Default is 0.0
    tmax
        Final time. Default is 2.0
    y0
        *(shape=(1, ))* -- Initial value. Default is ``[0.1]``.
    params
        Parameters :math:`(a, b)` for the logistic IVP.
        Default is :math:`(a, b) = (3.0, 1.0)`.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the logistic ODE with the prescribed
        configuration.
    """
    if y0 is None:
        y0 = np.array([0.1])

    def rhs(t, y, params=params):
        l0, l1 = params
        return l0 * y * (1.0 - y / l1)

    def jac(t, y, params=params):
        l0, l1 = params
        return np.array([l0 - l0 / l1 * 2 * y])

    def hess(t, y, params=params):
        l0, l1 = params
        return np.array([[-2 * l0 / l1]])

    def sol(t):
        l0, l1 = params
        nomin = l1 * y0 * np.exp(l0 * t)
        denom = l1 + y0 * (np.exp(l0 * t) - 1)
        return nomin / denom

    return InitialValueProblem(
        f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac, ddf=hess, solution=sol
    )


def fitzhughnagumo(t0=0.0, tmax=20.0, y0=None, params=(0.2, 0.2, 3.0, 1.0)):
    r"""Initial value problem (IVP) based on the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo (FHN) model is defined through

    .. math::

        f(t, y) =
        \begin{pmatrix}
            y_1 - \frac{1}{3} y_1^3 - y_2 + a \\
            \frac{1}{d} (y_1 + b - c y_2)
        \end{pmatrix}

    for some parameters :math:`(a, b, c, d)`.
    Default is :math:`(a, b)=(0.2, 0.2, 3.0)`.
    This implementation includes the Jacobian :math:`J_f` of :math:`f`.

    Parameters
    ----------
    t0
        Initial time. Default is 0.0
    tmax
        Final time. Default is 20.0
    y0
        *(shape=(2, ))* -- Initial value. Default is ``[1., -1.]``.
    params
        Parameter of the FitzHugh-Nagumo model. Default is ``(0.2, 0.2, 3.0, 1.0)``.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the FitzHugh-Nagumo model with the prescribed
        configuration.
    """
    if y0 is None:
        y0 = np.array([1.0, -1.0])

    def rhs(t, y, params=params):
        y1, y2 = y
        a, b, c, d = params
        return np.array([y1 - y1 ** 3.0 / 3.0 - y2 + a, (y1 + b - c * y2) / d])

    def jac(t, y, params=params):
        y1, y2 = y
        a, b, c, d = params
        return np.array([[1.0 - y1 ** 2.0, -1.0], [1.0 / d, -c / d]])

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def lotkavolterra(t0=0.0, tmax=20.0, y0=None, params=(0.5, 0.05, 0.5, 0.05)):
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
    t0
        Initial time. Default is 0.0
    tmax
        Final time. Default is 20.0
    y0
        *(shape=(2, ))* -- Initial value. Default is ``[1., -1.]``.
    params
        Parameter of the Lotka-Volterra model. Default is ``(0.2, 0.2, 3.0)``.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the Lotka-Volterra system with the prescribed
        configuration.
    """
    if y0 is None:
        y0 = np.array([20.0, 20.0])

    def rhs(t, y, params=params):
        a, b, c, d = params
        y1, y2 = y
        return np.array([a * y1 - b * y1 * y2, -c * y2 + d * y1 * y2])

    def jac(t, y, params=params):
        a, b, c, d = params
        y1, y2 = y
        return np.array([[a - b * y2, -b * y1], [d * y2, -c + d * y1]])

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def seir(t0=0.0, tmax=200.0, y0=None, params=(0.3, 0.3, 0.1)):
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
    t0
        Initial time. Default is 0.0
    tmax
        Final time. Default is 200.0
    y0
        *(shape=(4, ))* -- Initial value. Default is ``[998, 1, 1, 0]``.
    params
        Parameter of the SEIR model. Default is ``(0.3, 0.3, 0.1)``.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the SEIR model with the prescribed
        configuration.
    """
    if y0 is None:
        y0 = np.array([998, 1, 1, 0])

    params = params + (np.sum(y0),)

    def rhs(t, y, params=params):
        alpha, beta, gamma, population_count = params
        y1, y2, y3, y4 = y
        y1_next = -beta * y1 * y3 / population_count
        y2_next = beta * y1 * y3 / population_count - alpha * y2
        y3_next = alpha * y2 - gamma * y3
        y4_next = gamma * y3
        return np.array([y1_next, y2_next, y3_next, y4_next])

    def jac(t, y, params=params):
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

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def lorenz(t0=0.0, tmax=20.0, y0=None, params=(10.0, 28.0, 8.0 / 3.0)):
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
    t0
        Initial time. Default is 0.0
    tmax
        Final time. Default is 20.0
    y0
        *(shape=(3, ))* -- Initial value. Default is ``[0., 1., 1.05]``.
    params
        Parameter of the Lotka-Volterra model. Default is ``(0.2, 0.2, 3.0)``.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object describing the Lorenz system with the prescribed
        configuration.
    """
    if y0 is None:
        y0 = np.array([0.0, 1.0, 1.05])

    def rhs(t, y, params=params):
        a, b, c = params
        y1, y2, y3 = y
        return np.array([a * (y2 - y1), y1 * (b - y3) - y2, y1 * y2 - c * y3])

    def jac(t, y, params=params):
        a, b, c = params
        y1, y2, y3 = y
        return np.array([[-a, a, 0], [b - y3, -1, -y1], [y2, y1, -c]])

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)
