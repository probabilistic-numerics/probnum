"""Initialisation procedures."""


import functools

import numpy as np

import probnum.diffeq as pnd
import probnum.random_variables as pnrv

__all__ = ["compute_all_derivatives"]


def compute_all_derivatives(ivp, order=11):
    """Fill dy0_all field of InitialValueProblem.

    Examples
    --------
    >>> from probnum.problems.zoo.diffeq import threebody_jax
    >>> ivp = threebody_jax()
    >>> print(ivp.y0)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(ivp.dy0_all)
    None
    >>> ivp2 = compute_all_derivatives(ivp, order=3)
    >>> print(ivp2.y0)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(ivp2.dy0_all)
    [ 9.94000000e-01  0.00000000e+00  0.00000000e+00 -2.00158511e+00
      0.00000000e+00 -2.00158511e+00 -3.15543023e+02  0.00000000e+00
     -3.15543023e+02  0.00000000e+00  0.00000000e+00  9.99720945e+04
      0.00000000e+00  9.99720945e+04  6.39028111e+07  0.00000000e+00]
    """
    ivp.dy0_all = _taylormode(f=ivp.f, z0=ivp.y0, t0=ivp.t0, order=order)
    return ivp


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
    except ImportError:
        raise ImportError("Initialisation requires jax. Sorry... :( ")

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
