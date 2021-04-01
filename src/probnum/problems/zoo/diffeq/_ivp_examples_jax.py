import numpy as np

from probnum.problems import InitialValueProblem

__all__ = ["threebody_jax", "vanderpol_jax"]
# pylint: disable=import-outside-toplevel


JAX_ERRORMSG = "IVP instantiation requires jax. Try using the pure numpy versions instead, or install `jax` via `pip install jax jaxlib`"


def threebody_jax(tmax=17.0652165601579625588917206249):
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
    tmax
        Final time.

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

    try:
        import jax
        import jax.numpy as jnp
        from jax.config import config

        config.update("jax_enable_x64", True)

    except ImportError as err:
        raise ImportError(JAX_ERRORMSG) from err

    def threebody_rhs(Y):
        # defining the ODE:
        # assume Y = [y1,y2,y1',y2']
        mu = 0.012277471  # a constant (standardised moon mass)
        mp = 1 - mu
        D1 = ((Y[0] + mu) ** 2 + Y[1] ** 2) ** (3 / 2)
        D2 = ((Y[0] - mp) ** 2 + Y[1] ** 2) ** (3 / 2)
        y1p = Y[0] + 2 * Y[3] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        y2p = Y[1] - 2 * Y[2] - mp * Y[1] / D1 - mu * Y[1] / D2
        return jnp.array([Y[2], Y[3], y1p, y2p])

    df = jax.jacfwd(threebody_rhs)
    ddf = jax.jacrev(df)

    def rhs(t, y):
        return threebody_rhs(Y=y)

    def jac(t, y):
        return df(y)

    def hess(t, y):
        return ddf(y)

    y0 = np.array([0.994, 0, 0, -2.00158510637908252240537862224])
    t0 = 0.0
    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac, ddf=hess)


def vanderpol_jax(t0=0.0, tmax=30, y0=None, params=1e1):
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
    IVP
        IVP object describing the Van der Pol Oscillator IVP with the prescribed
        configuration.
    """

    try:
        import jax
        import jax.numpy as jnp
        from jax.config import config

        config.update("jax_enable_x64", True)

    except ImportError as err:
        raise ImportError(JAX_ERRORMSG) from err

    if isinstance(params, float):
        mu = params
    else:
        (mu,) = params

    if y0 is None:
        y0 = np.array([2.0, 0.0])

    def vanderpol_rhs(Y):
        return jnp.array([Y[1], mu * (1.0 - Y[0] ** 2) * Y[1] - Y[0]])

    df = jax.jacfwd(vanderpol_rhs)
    ddf = jax.jacrev(df)

    def rhs(t, y):
        return vanderpol_rhs(Y=y)

    def jac(t, y):
        return df(y)

    def hess(t, y):
        return ddf(y)

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac, ddf=hess)
