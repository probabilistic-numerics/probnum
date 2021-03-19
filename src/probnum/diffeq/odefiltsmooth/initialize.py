"""Initialisation procedures."""
# pylint: disable=import-outside-toplevel


import numpy as np
import scipy.integrate as sci

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum import randvars

# In the initialisation-via-RK function below, this value is added to the marginal stds of the initial derivatives that are known.
# If we put in zero, there are linalg errors (because a zero-cov RV is conditioned on a dirac likelihood).
# This value is chosen such that its square-root is a really small damping factor).
SMALL_VALUE = 1e-28


def initialize_odefilter_with_rk(
    f, y0, t0, prior, initrv, df=None, h0=1e-2, method="DOP853"
):
    r"""Initialize an ODE filter by fitting the prior process to a few steps of an approximate ODE solution computed with Scipy's RK.

    It goes as follows:

    1. The ODE integration problem is set up on the interval ``[t0, t0 + (2*order+1)*h0]``
    and solved with a call to ``scipy.integrate.solve_ivp``. The solver is uses adaptive steps with ``atol=rtol=1e-12``,
    but is forced to pass through the
    events ``(t0, t0+h0, t0 + 2*h0, ..., t0 + (2*order+1)*h0)``.
    The result is a vector of time points and states, with at least ``(2*order+1)``.
    Potentially, the adaptive steps selected many more steps, but because of the events, fewer steps cannot have happened.

    2. A prescribed prior is fitted to the first ``(2*order+1)`` (t, y) pairs of the solution. ``order`` is the order of the prior.

    3. The value of the resulting posterior at time ``t=t0`` is an estimate of the state and all its derivatives.
    The resulting marginal standard deviations estimate the error. This random variable is returned.

    Parameters
    ----------
    f
        ODE vector field.
    y0
        Initial value.
    t0
        Initial time point.
    prior
        Prior distribution used for the ODE solver. For instance an integrated Brownian motion prior (``IBM``).
    initrv
        Initial random variable.
    df
        Jacobian of the ODE vector field. Optional. If specified, more components of the result will be exact.
    h0
        Maximum step-size to use for computing the approximate ODE solution. The smaller, the more accurate, but also, the smaller, the less stable.
        The best value here depends on the ODE problem, and probably the chosen method. Optional. Default is ``1e-2``.
    method
        Which solver to use. This is communicated as a string that is compatible with ``scipy.integrate.solve_ivp(..., method=method)``.
        Optional. Default is `DOP853`.

    Returns
    -------
    Normal
        Estimated (improved) initial random variable. Compatible with the specified prior.

    Examples
    --------

    >>> from dataclasses import astuple
    >>> from probnum.randvars import Normal
    >>> from probnum.statespace import IBM
    >>> from probnum.problems.zoo.diffeq import vanderpol

    Compute the initial values of the van-der-Pol problem as follows

    >>> f, t0, tmax, y0, df, *_ = astuple(vanderpol())
    >>> print(y0)
    [2. 0.]
    >>> prior = IBM(ordint=3, spatialdim=2)
    >>> initrv = Normal(mean=np.zeros(prior.dimension), cov=np.eye(prior.dimension))
    >>> improved_initrv = initialize_odefilter_with_rk(f, y0, t0, prior=prior, initrv=initrv, df=df)
    >>> print(prior.proj2coord(0) @ improved_initrv.mean)
    [2. 0.]
    >>> print(np.round(improved_initrv.mean, 1))
    [    2.      0.     -2.     58.2     0.     -2.     60.  -1745.7]
    >>> print(np.round(np.log10(improved_initrv.std), 1))
    [-13.8 -11.3  -9.   -1.5 -13.8 -11.3  -9.   -1.5]
    """
    y0 = np.asarray(y0)
    ode_dim = y0.shape[0] if y0.ndim > 0 else 1
    order = prior.ordint
    proj_to_y = prior.proj2coord(0)
    zeros_shift = np.zeros(ode_dim)
    zeros_cov = np.zeros((ode_dim, ode_dim))
    measmod = pnss.DiscreteLTIGaussian(
        proj_to_y,
        zeros_shift,
        zeros_cov,
        proc_noise_cov_cholesky=zeros_cov,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )

    # order + 1 would suffice in theory, 2*order + 1 is for good measure
    # (the "+1" is a safety factor for order=1)
    num_steps = 2 * order + 1
    t_eval = np.arange(t0, t0 + (num_steps + 1) * h0, h0)
    sol = sci.solve_ivp(
        f,
        (t0, t0 + (num_steps + 1) * h0),
        y0=y0,
        atol=1e-12,
        rtol=1e-12,
        t_eval=t_eval,
        method=method,
    )

    ts = sol.t[:num_steps]
    ys = sol.y[:, :num_steps].T

    initmean = initrv.mean.copy()
    initmean[0 :: (order + 1)] = y0
    initmean[1 :: (order + 1)] = f(t0, y0)

    initcov_diag = np.diag(initrv.cov).copy()
    initcov_diag[0 :: (order + 1)] = SMALL_VALUE
    initcov_diag[1 :: (order + 1)] = SMALL_VALUE

    if df is not None:
        if order > 1:
            initmean[2 :: (order + 1)] = df(t0, y0) @ f(t0, y0)
            initcov_diag[2 :: (order + 1)] = SMALL_VALUE

    initcov = np.diag(initcov_diag)
    initcov_cholesky = np.diag(np.sqrt(initcov_diag))
    initrv = randvars.Normal(initmean, initcov, cov_cholesky=initcov_cholesky)
    kalman = pnfs.Kalman(prior, measmod, initrv)

    out = kalman.filtsmooth(ys, ts)

    estimated_initrv = out.state_rvs[0]

    return estimated_initrv


def initialize_odefilter_with_taylormode(f, y0, t0, prior, initrv):
    """Initialize an ODE filter with Taylor-mode automatic differentiation.

    This requires JAX. For an explanation of what happens ``under the hood``, see [1]_.

    References
    ----------
    .. [1] Krämer, N. and Hennig, P., Stable implementation of probabilistic ODE solvers,
       *arXiv:2012.10106*, 2020.


    The implementation is inspired by the implementation in
    https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py

    Parameters
    ----------
    f
        ODE vector field.
    y0
        Initial value.
    t0
        Initial time point.
    prior
        Prior distribution used for the ODE solver. For instance an integrated Brownian motion prior (``IBM``).
    initrv
        Initial random variable.

    Returns
    -------
    Normal
        Estimated initial random variable. Compatible with the specified prior.


    Examples
    --------

    >>> import sys, pytest
    >>> if sys.platform.startswith('win'):
    ...     pytest.skip('this doctest does not work on Windows')

    >>> from dataclasses import astuple
    >>> from probnum.randvars import Normal
    >>> from probnum.problems.zoo.diffeq import threebody_jax, vanderpol_jax
    >>> from probnum.statespace import IBM

    Compute the initial values of the restricted three-body problem as follows

    >>> f, t0, tmax, y0, df, *_ = astuple(threebody_jax())
    >>> print(y0)
    [ 0.994       0.          0.         -2.00158511]

    >>> prior = IBM(ordint=3, spatialdim=4)
    >>> initrv = Normal(mean=np.zeros(prior.dimension), cov=np.eye(prior.dimension))
    >>> improved_initrv = initialize_odefilter_with_taylormode(f, y0, t0, prior, initrv)
    >>> print(prior.proj2coord(0) @ improved_initrv.mean)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(improved_initrv.mean)
    [ 9.94000000e-01  0.00000000e+00 -3.15543023e+02  0.00000000e+00
      0.00000000e+00 -2.00158511e+00  0.00000000e+00  9.99720945e+04
      0.00000000e+00 -3.15543023e+02  0.00000000e+00  6.39028111e+07
     -2.00158511e+00  0.00000000e+00  9.99720945e+04  0.00000000e+00]

    Compute the initial values of the van-der-Pol oscillator as follows

    >>> f, t0, tmax, y0, df, *_ = astuple(vanderpol_jax())
    >>> print(y0)
    [2. 0.]
    >>> prior = IBM(ordint=3, spatialdim=2)
    >>> initrv = Normal(mean=np.zeros(prior.dimension), cov=np.eye(prior.dimension))
    >>> improved_initrv = initialize_odefilter_with_taylormode(f, y0, t0, prior, initrv)
    >>> print(prior.proj2coord(0) @ improved_initrv.mean)
    [2. 0.]
    >>> print(improved_initrv.mean)
    [    2.     0.    -2.    60.     0.    -2.    60. -1798.]
    >>> print(improved_initrv.std)
    [0. 0. 0. 0. 0. 0. 0. 0.]
    """

    try:
        import jax.numpy as jnp
        from jax.config import config
        from jax.experimental.jet import jet

        config.update("jax_enable_x64", True)
    except ImportError as err:
        raise ImportError(
            "Cannot perform Taylor-mode initialisation without optional "
            "dependencies jax and jaxlib. Try installing them via `pip install jax jaxlib`."
        ) from err

    order = prior.ordint

    def total_derivative(z_t):
        """Total derivative."""
        z, t = jnp.reshape(z_t[:-1], z_shape), z_t[-1]
        dz = jnp.ravel(f(t, z))
        dt = jnp.array([1.0])
        dz_t = jnp.concatenate((dz, dt))
        return dz_t

    z_shape = y0.shape
    z_t = jnp.concatenate((jnp.ravel(y0), jnp.array([t0])))

    derivs = []

    derivs.extend(y0)
    if order == 0:
        all_derivs = pnss.Integrator._convert_derivwise_to_coordwise(
            np.asarray(jnp.array(derivs)), ordint=0, spatialdim=len(y0)
        )

        return randvars.Normal(
            np.asarray(all_derivs),
            cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
            cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        )

    (dy0, [*yns]) = jet(total_derivative, (z_t,), ((jnp.ones_like(z_t),),))
    derivs.extend(dy0[:-1])
    if order == 1:
        all_derivs = pnss.Integrator._convert_derivwise_to_coordwise(
            np.asarray(jnp.array(derivs)), ordint=1, spatialdim=len(y0)
        )

        return randvars.Normal(
            np.asarray(all_derivs),
            cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
            cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        )

    for _ in range(1, order):
        (dy0, [*yns]) = jet(total_derivative, (z_t,), ((dy0, *yns),))
        derivs.extend(yns[-2][:-1])

    all_derivs = pnss.Integrator._convert_derivwise_to_coordwise(
        jnp.array(derivs), ordint=order, spatialdim=len(y0)
    )

    return randvars.Normal(
        np.asarray(all_derivs),
        cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
    )
