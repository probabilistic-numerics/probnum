"""Initialisation procedures."""
# pylint: disable=import-outside-toplevel


import numpy as np
import scipy.integrate as sci

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from probnum.filtsmooth import statespace as pnss
from probnum.problems import InitialValueProblem

__all__ = [
    "extend_ivp_with_all_derivatives",
    "compute_all_derivatives_via_rk",
    "compute_all_derivatives_via_taylormode",
]

# pylint: disable=import-outside-toplevel


def extend_ivp_with_all_derivatives(ivp, order=6):
    r"""Compute derivatives of initial values of an initial value ODE problem.

    This requires jax. For an explanation of what happens "under the hood", see [1]_.


    Parameters
    ----------
    ivp: InitialValueProblem
        Initial value problem. See `probnum.problems`.
    order : int
        How many derivatives are required? Optional, default is 6. This will later be referred to as :math:`\nu`.

    Raises
    ------
    ImportError
        If `jax` or `jaxlib` are not available.

    Returns
    -------
    InitialValueProblem
        InitialValueProblem object where the attribute `dy0_all` is filled with an
        :math:`d(\nu + 1)` vector, where :math:`\nu` is the specified order,
        and :math:`d` is the dimension of the IVP.

    References
    ----------
    .. [1] Krämer, N. and Hennig, P., Stable implementation of probabilistic ODE solvers,
       *arXiv:2012.10106*, 2020.


    Examples
    --------
    >>> from probnum.problems.zoo.diffeq import threebody_jax, vanderpol_jax, threebody, vanderpol

    Compute the initial values of the restricted three-body problem as follows

    >>> res2bod = threebody_jax()
    >>> print(res2bod.y0)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(res2bod.dy0_all)
    None
    >>> res2bod = compute_all_derivatives(res2bod, order=3)
    >>> print(res2bod.y0)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(res2bod.dy0_all)
    [ 9.94000000e-01  0.00000000e+00 -3.15543023e+02  0.00000000e+00
      0.00000000e+00 -2.00158511e+00  0.00000000e+00  9.99720945e+04
      0.00000000e+00 -3.15543023e+02  0.00000000e+00  6.39028111e+07
     -2.00158511e+00  0.00000000e+00  9.99720945e+04  0.00000000e+00]

    Compute the initial values of the van-der-Pol oscillator as follows

    >>> vdp = vanderpol_jax()
    >>> print(vdp.y0)
    [2. 0.]
    >>> print(vdp.dy0_all)
    None
    >>> vdp = extend_ivp_with_all_derivatives(vdp, order=3)
    >>> print(vdp.y0)
    [2. 0.]
    >>> print(vdp.dy0_all)
    [    2.     0.    -2.    60.     0.    -2.    60. -1798.]


    >>> vdp2 = vanderpol()
    >>> print(vdp2.y0)
    [2. 0.]
    >>> print(vdp2.dy0_all)
    None
    >>> vdp2 = extend_ivp_with_all_derivatives(vdp2, order=3)
    >>> print(vdp2.y0)
    [2. 0.]
    >>> print(np.round(vdp2.dy0_all, 1))
    [    2.     -0.     -2.     49.7     0.     -2.     59.  -1492. ]
    >>> print(np.round(np.log10(vdp2.dy0_errors), 1))
    [-inf -6.1 -3.5 -1.1 -inf -6.1 -3.5 -1.1]
    """
    try:
        all_initial_derivatives, errors = compute_all_derivatives_via_taylormode(
            f=ivp.f, z0=ivp.y0, t0=ivp.t0, order=order
        )
        all_initial_derivatives = _correct_order_of_elements(
            all_initial_derivatives, order
        )
    except KeyError:
        all_initial_derivatives, errors = compute_all_derivatives_via_rk(
            f=ivp.f, z0=ivp.y0, t0=ivp.t0, order=order, df=ivp.df
        )

    return InitialValueProblem(
        f=ivp.f,
        t0=ivp.t0,
        tmax=ivp.tmax,
        y0=ivp.y0,
        df=ivp.df,
        ddf=ivp.ddf,
        solution=ivp.solution,
        dy0_all=all_initial_derivatives,
        dy0_errors=errors,
    )


def _correct_order_of_elements(arr, order):
    """Utility function to change ordering of elements in stacked vector."""
    return arr.reshape((order + 1, -1)).T.flatten()


SMALL_VALUE = 1e-28


def compute_all_derivatives_via_rk(f, z0, t0, order, df=None, h0=1e-2, method="DOP853"):
    """Solve the ODE for a few steps with scipy.integrate, and fit an integrated Wiener
    process to the solution.

    The resulting value at t0 is an estimate of the initial derivatives.
    """
    ode_dim = len(z0)
    prior = pnss.IBM(
        ordint=order,
        spatialdim=ode_dim,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
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

    num_steps = (
        2 * order
    )  # order + 1 would suffice in theory, 2*order is for good measure
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
    mean = estimated_initrv.mean
    std = estimated_initrv.std
    return mean, std


def compute_all_derivatives_via_taylormode(f, z0, t0, order):
    """Taylor-mode automatic differentiation for initialisation.

    Inspired by the implementation in
    https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
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
        return jnp.array(derivs), jnp.zeros(len(derivs))
    (y0, [*yns]) = jet(total_derivative, (z_t,), ((jnp.ones_like(z_t),),))
    derivs.extend(y0[:-1])
    if order == 1:
        return jnp.array(derivs), jnp.zeros(len(derivs))

    order = order - 2
    for _ in range(order + 1):
        (y0, [*yns]) = jet(total_derivative, (z_t,), ((y0, *yns),))
        derivs.extend(yns[-2][:-1])

    return jnp.array(derivs), jnp.zeros(len(derivs))
