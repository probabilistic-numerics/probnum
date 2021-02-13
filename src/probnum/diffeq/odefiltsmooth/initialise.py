"""Initialisation procedures."""


import functools

import numpy as np

import probnum.diffeq as pnd
import probnum.random_variables as pnrv

__all__ = ["compute_all_derivatives"]


def compute_all_derivatives(ivp, order=6):
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
    >>> from probnum.problems.zoo.diffeq import threebody_jax, vanderpol_jax

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
    [ 9.94000000e-01  0.00000000e+00  0.00000000e+00 -2.00158511e+00
      0.00000000e+00 -2.00158511e+00 -3.15543023e+02  0.00000000e+00
     -3.15543023e+02  0.00000000e+00  0.00000000e+00  9.99720945e+04
      0.00000000e+00  9.99720945e+04  6.39028111e+07  0.00000000e+00]

    Compute the initial values of the van-der-Pol oscillator as follows

    >>> vdp = vanderpol_jax()
    >>> print(vdp.y0)
    [2. 0.]
    >>> print(vdp.dy0_all)
    None
    >>> vdp = compute_all_derivatives(vdp, order=3)
    >>> print(vdp.y0)
    [2. 0.]
    >>> print(vdp.dy0_all)
    [    2.     0.     0.    -2.    -2.    60.    60. -1798.]
    """
    all_initial_derivatives = _taylormode(f=ivp.f, z0=ivp.y0, t0=ivp.t0, order=order)
    return InitialValueProblem(
        f=ivp.f,
        t0=ivp.t0,
        tmax=ivp.tmax,
        y0=ivp.y0,
        df=ivp.df,
        ddf=ivp.ddf,
        solution=ivp.solution,
        dy0_all=all_initial_derivatives,
    )


def _taylormode(f, z0, t0, order):
    """Taylor-mode automatic differentiation for initialisation.

    Inspired by the implementation in
    https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
    """

    try:
        import jax
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
        return jnp.array(derivs)
    (y0, [*yns]) = jet(total_derivative, (z_t,), ((jnp.ones_like(z_t),),))
    derivs.extend(y0[:-1])
    if order == 1:
        return jnp.array(derivs)

    order = order - 2
    for _ in range(order + 1):
        (y0, [*yns]) = jet(total_derivative, (z_t,), ((y0, *yns),))
        derivs.extend(yns[-2][:-1])

    return jnp.array(derivs)
