"""Initialisation procedures."""
# pylint: disable=import-outside-toplevel


import numpy as np
import scipy.integrate as sci

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnss

__all__ = [
    "initialize_odefilter_with_rk",
    "initialize_odefilter_with_taylormode",
]


# pylint: disable=import-outside-toplevel
#
#
# def extend_ivp_with_all_derivatives(ivp, order):
#     r"""Create a new InitialValueProblem which is informed about derivatives of initial conditions.
#
#     If the InitialValueProblem is compatible with JAX (and if jax and jaxlib can be imported), this is done with
#     Taylor-mode automatic differentiation.
#     If not, an integrated Brownian motion process is fitted to the first few states of an ODE solution computed with a Runge-Kutta formula.
#
#
#
#     Parameters
#     ----------
#     ivp: InitialValueProblem
#         Initial value problem. See `probnum.problems`.
#     order : int
#         How many derivatives are required? Optional, default is 6. This will later be referred to as :math:`\nu`.
#
#     Raises
#     ------
#     ImportError
#         If `jax` or `jaxlib` are not available.
#
#     Returns
#     -------
#     InitialValueProblem
#         InitialValueProblem object where the attribute `dy0_all` is filled with an
#         :math:`d(\nu + 1)` vector, where :math:`\nu` is the specified order,
#         and :math:`d` is the dimension of the IVP.
#
#
#     Examples
#     --------
#     >>> import sys, pytest
#     >>> if sys.platform.startswith('win'):
#     ...     pytest.skip('this doctest does not work on Windows')
#
#     >>> from probnum.problems.zoo.diffeq import threebody_jax, vanderpol_jax, threebody, vanderpol
#
#     Compute the initial values of the restricted three-body problem as follows
#
#     >>> res2bod = threebody_jax()
#     >>> print(res2bod.y0)
#     [ 0.994       0.          0.         -2.00158511]
#     >>> print(res2bod.dy0_all)
#     None
#     >>> res2bod = extend_ivp_with_all_derivatives(res2bod, order=3)
#     >>> print(res2bod.y0)
#     [ 0.994       0.          0.         -2.00158511]
#     >>> print(res2bod.dy0_all)
#     [ 9.94000000e-01  0.00000000e+00 -3.15543023e+02  0.00000000e+00
#       0.00000000e+00 -2.00158511e+00  0.00000000e+00  9.99720945e+04
#       0.00000000e+00 -3.15543023e+02  0.00000000e+00  6.39028111e+07
#      -2.00158511e+00  0.00000000e+00  9.99720945e+04  0.00000000e+00]
#
#     Compute the initial values of the van-der-Pol oscillator as follows
#
#     >>> vdp = vanderpol_jax()
#     >>> print(vdp.y0)
#     [2. 0.]
#     >>> print(vdp.dy0_all)
#     None
#     >>> vdp = extend_ivp_with_all_derivatives(vdp, order=3)
#     >>> print(vdp.y0)
#     [2. 0.]
#     >>> print(vdp.dy0_all)
#     [    2.     0.    -2.    60.     0.    -2.    60. -1798.]
#
#
#     >>> vdp2 = vanderpol()
#     >>> print(vdp2.y0)
#     [2. 0.]
#     >>> print(vdp2.dy0_all)
#     None
#     >>> vdp2 = extend_ivp_with_all_derivatives(vdp2, order=3)
#     >>> print(vdp2.y0)
#     [2. 0.]
#     >>> print(np.round(vdp2.dy0_all, 1))
#     [    2.      0.     -2.     58.2     0.     -2.     60.  -1743.4]
#     >>> print(np.round(np.log10(vdp2.dy0_errors), 1))
#     [ -inf -14.  -14.   -1.5  -inf -14.  -14.   -1.5]
#     """
#     try:
#         all_initial_derivatives, errors = initialize_odefilter_with_taylormode(
#             f=ivp.f, z0=ivp.y0, t0=ivp.t0, order=order
#         )
#     except KeyError:
#         all_initial_derivatives, errors = initialize_odefilter_with_rk(
#             f=ivp.f, z0=ivp.y0, t0=ivp.t0, order=order, df=ivp.df
#         )
#
#     return InitialValueProblem(
#         f=ivp.f,
#         t0=ivp.t0,
#         tmax=ivp.tmax,
#         y0=ivp.y0,
#         df=ivp.df,
#         ddf=ivp.ddf,
#         solution=ivp.solution,
#         dy0_all=all_initial_derivatives,
#         dy0_errors=errors,
#     )


SMALL_VALUE = 1e-28


def initialize_odefilter_with_rk(f, z0, t0, prior, df=None, h0=1e-2, method="DOP853"):
    r"""Compute derivatives of the initial value by fitting an integrated Brownian motion process to a few steps of an approximate ODE solution.


    It goes as follows:

    1. The ODE integration problem is set up on the interval ``[t0, t0 + (2*order+1)*h0]``
    and solved with a call to ``scipy.integrate.solve_ivp``. The solver is uses adaptive steps with ``atol=rtol=1e-12``,
    but is forced to pass through the
    events ``(t0, t0+h0, t0 + 2*h0, ..., t0 + (2*order+1)*h0)``.
    The result is a vector of time points and states, with at least ``(2*order+1)``.
    Potentially, the adaptive steps selected many more steps, but because of the events, fewer steps cannot have happened.

    2. A :math:`\nu` times integrated Brownian motion process is fitted to the first ``(2*order+1)`` (t, y) pairs of the solution.

    3. The value of the resulting posterior at time ``t=t0`` is an estimate of the state and all its derivatives.
    The resulting marginal standard deviations estimate the error.

    Parameters
    ----------
    f
        ODE vector field.
    z0
        Initial value.
    t0
        Initial time point.
    order
        Number of derivatives to compute. Corresponds to the order of the prior (for instance the number of times an integrated Brownian motion process is integrated).
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
    array
        Estimate of the full stack of derivatives
    array
        Marginal standard deviations of the state, can be used as an error estimate of the estimation.

    Examples
    --------

    >>> from dataclasses import astuple
    >>> from probnum.filtsmooth.statespace import IBM
    >>> from probnum.problems.zoo.diffeq import vanderpol

    Compute the initial values of the restricted three-body problem as follows

    >>> f, t0, tmax, y0, df, *_ = astuple(vanderpol())
    >>> print(y0)
    [2. 0.]
    >>> prior = IBM(ordint=3, spatialdim=2)
    >>> initrv = initialize_odefilter_with_rk(f, y0, t0, prior=prior, df=df)
    >>> print(prior.proj2coord(0) @ initrv.mean)
    [2. 0.]
    >>> print(np.round(initrv.mean, 1))
    [    2.      0.     -2.     58.2     0.     -2.     60.  -1745.7]
    >>> print(np.round(np.log10(initrv.std), 1))
    [-13.8 -11.3  -9.   -1.5 -13.8 -11.3  -9.   -1.5]
    """
    z0 = np.asarray(z0)
    ode_dim = z0.shape[0] if z0.ndim > 0 else 1
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
        y0=z0,
        atol=1e-12,
        rtol=1e-12,
        t_eval=t_eval,
        method=method,
    )

    ts = sol.t[:num_steps]
    ys = sol.y[:, :num_steps].T

    initmean = np.zeros(prior.dimension)
    initmean[0 :: (order + 1)] = z0
    initmean[1 :: (order + 1)] = f(t0, z0)

    initcov_diag = np.ones(prior.dimension)
    initcov_diag[0 :: (order + 1)] = SMALL_VALUE
    initcov_diag[1 :: (order + 1)] = SMALL_VALUE

    if df is not None:
        if order > 1:
            initmean[2 :: (order + 1)] = df(t0, z0) @ f(t0, z0)
            initcov_diag[2 :: (order + 1)] = SMALL_VALUE

    initcov = np.diag(initcov_diag)
    initcov_cholesky = np.diag(np.sqrt(initcov_diag))
    initrv = pnrv.Normal(initmean, initcov, cov_cholesky=initcov_cholesky)
    kalman = pnfs.Kalman(prior, measmod, initrv)

    out = kalman.filtsmooth(ys, ts)

    estimated_initrv = out.state_rvs[0]
    return estimated_initrv


def initialize_odefilter_with_taylormode(f, z0, t0, prior):
    """Compute derivatives of the initial conditions of an IVP with Taylor-mode
    automatic differentiation.

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
    z0
        Initial value.
    t0
        Initial time point.
    order
        Number of derivatives to compute. Corresponds to the order of the prior (for instance the number of times an integrated Brownian motion process is integrated).

    Returns
    -------
    array
        Estimate of the full stack of derivatives
    array
        Array full of zeros, as a dummy error estimate of the estimation.


    Examples
    --------

    >>> import sys, pytest
    >>> if sys.platform.startswith('win'):
    ...     pytest.skip('this doctest does not work on Windows')

    >>> from dataclasses import astuple
    >>> from probnum.problems.zoo.diffeq import threebody_jax, vanderpol_jax
    >>> from probnum.filtsmooth.statespace import IBM

    Compute the initial values of the restricted three-body problem as follows

    >>> f, t0, tmax, y0, df, *_ = astuple(threebody_jax())
    >>> print(y0)
    [ 0.994       0.          0.         -2.00158511]

    >>> prior = IBM(ordint=3, spatialdim=4)
    >>> initrv = initialize_odefilter_with_taylormode(f, y0, t0, prior)
    >>> print(prior.proj2coord(0) @ initrv.mean)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(initrv.mean)
    [ 9.94000000e-01  0.00000000e+00 -3.15543023e+02  0.00000000e+00
      0.00000000e+00 -2.00158511e+00  0.00000000e+00  9.99720945e+04
      0.00000000e+00 -3.15543023e+02  0.00000000e+00  6.39028111e+07
     -2.00158511e+00  0.00000000e+00  9.99720945e+04  0.00000000e+00]

    Compute the initial values of the van-der-Pol oscillator as follows

    >>> f, t0, tmax, y0, df, *_ = astuple(vanderpol_jax())
    >>> print(y0)
    [2. 0.]
    >>> prior = IBM(ordint=3, spatialdim=2)
    >>> initrv = initialize_odefilter_with_taylormode(f, y0, t0, prior)
    >>> print(prior.proj2coord(0) @ initrv.mean)
    [2. 0.]
    >>> print(initrv.mean)
    [    2.     0.    -2.    60.     0.    -2.    60. -1798.]
    >>> print(initrv.std)
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

    z_shape = z0.shape
    z_t = jnp.concatenate((jnp.ravel(z0), jnp.array([t0])))

    derivs = []

    derivs.extend(z0)
    if order == 0:
        all_derivs = pnss.Integrator._convert_derivwise_to_coordwise(
            jnp.array(derivs), ordint=0, spatialdim=len(z0)
        )

        return pnrv.Normal(
            np.asarray(all_derivs),
            cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
            cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        )

    (y0, [*yns]) = jet(total_derivative, (z_t,), ((jnp.ones_like(z_t),),))
    derivs.extend(y0[:-1])
    if order == 1:
        all_derivs = pnss.Integrator._convert_derivwise_to_coordwise(
            jnp.array(derivs), ordint=1, spatialdim=len(z0)
        )

        return pnrv.Normal(
            np.asarray(all_derivs),
            cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
            cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        )

    order = order - 2
    for _ in range(order + 1):
        (y0, [*yns]) = jet(total_derivative, (z_t,), ((y0, *yns),))
        derivs.extend(yns[-2][:-1])

    all_derivs = pnss.Integrator._convert_derivwise_to_coordwise(
        jnp.array(derivs), ordint=order + 2, spatialdim=len(z0)
    )

    return pnrv.Normal(
        np.asarray(all_derivs),
        cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
    )
