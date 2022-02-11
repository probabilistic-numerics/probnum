"""Automatic-differentiation-based initialization routines."""

import numpy as np

from probnum import problems, randprocs, randvars

from ._interface import InitializationRoutine

# pylint: disable="import-outside-toplevel"
try:
    import jax
    import jax.numpy as jnp
    from jax.config import config
    from jax.experimental.jet import jet

    config.update("jax_enable_x64", True)

    JAX_IS_AVAILABLE = True
except ImportError as JAX_IMPORT_ERROR:
    JAX_IS_AVAILABLE = False
    JAX_IMPORT_ERROR_MSG = (
        "Cannot perform Jax-based initialization without the optional "
        "dependencies jax and jaxlib. "
        "Try installing them via `pip install jax jaxlib`."
    )


class _AutoDiffBase(InitializationRoutine):
    def __init__(self):

        if not JAX_IS_AVAILABLE:
            raise ImportError(JAX_IMPORT_ERROR_MSG) from JAX_IMPORT_ERROR

        super().__init__(is_exact=True, requires_jax=True)

    def __call__(
        self,
        *,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:

        num_derivatives = prior_process.transition.num_derivatives

        f, y0 = self._make_autonomous(ivp=ivp)

        mean_matrix = self._compute_ode_derivatives(
            f=f, y0=y0, num_derivatives=num_derivatives
        )
        mean = mean_matrix.reshape((-1,), order="F")
        zeros = jnp.zeros((mean.shape[0], mean.shape[0]))
        return randvars.Normal(
            mean=np.asarray(mean),
            cov=np.asarray(zeros),
            cov_cholesky=np.asarray(zeros),
        )

    def _compute_ode_derivatives(self, *, f, y0, num_derivatives):
        gen = self._initial_derivative_generator(f=f, y0=y0)
        mean_matrix = jnp.stack(
            [next(gen)(y0)[:-1] for _ in range(num_derivatives + 1)]
        )
        return mean_matrix

    def _make_autonomous(self, *, ivp):
        """Preprocess the ODE.

        Turn the ODE into a format that is more convenient to handle with automatic
        differentiation. This has no effect on the ODE itself. It is purely internal.
        """
        y0_autonomous = jnp.concatenate([ivp.y0, jnp.array([ivp.t0])])

        def f_autonomous(y):
            x, t = y[:-1], y[-1]
            fx = ivp.f(t, x)
            return jnp.concatenate([fx, jnp.array([1.0])])

        return f_autonomous, y0_autonomous

    def _initial_derivative_generator(self, *, f, y0):
        """Generate the inital derivatives recursively."""

        def fwd_deriv(f, f0):
            def df(x):
                return self._jvp_or_vjp(fun=f, primals=x, tangents=f0(x))

            return df

        yield lambda x: y0

        g = f
        f0 = f
        while True:
            yield g
            g = fwd_deriv(g, f0)

    def _jvp_or_vjp(self, *, fun, primals, tangents):
        raise NotImplementedError


class ForwardModeJVP(_AutoDiffBase):
    """Initialization via Jacobian-vector-product-based automatic differentiation."""

    def _jvp_or_vjp(self, *, fun, primals, tangents):
        _, y = jax.jvp(fun, (primals,), (tangents,))
        return y


class ForwardMode(_AutoDiffBase):
    """Initialization via forward-mode automatic differentiation."""

    def _jvp_or_vjp(self, *, fun, primals, tangents):
        return jax.jacfwd(fun)(primals) @ tangents


class ReverseMode(_AutoDiffBase):
    """Initialization via reverse-mode automatic differentiation."""

    def _jvp_or_vjp(self, *, fun, primals, tangents):
        return jax.jacrev(fun)(primals) @ tangents


class TaylorMode(_AutoDiffBase):
    """Initialize a probabilistic ODE solver with Taylor-mode automatic differentiation.

    This requires JAX. For an explanation of what happens ``under the hood``, see [1]_.

    The implementation is inspired by the implementation in
    https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
    See also [2]_.

    References
    ----------
    .. [1] Krämer, N. and Hennig, P.,
       Stable implementation of probabilistic ODE solvers,
       *arXiv:2012.10106*, 2020.
    .. [2] Kelly, J. and Bettencourt, J. and Johnson, M. and Duvenaud, D.,
        Learning differential equations that are easy to solve,
        Neurips 2020.

    Examples
    --------

    >>> import sys, pytest
    >>> if not sys.platform.startswith('linux'):
    ...     pytest.skip()

    >>> import numpy as np
    >>> from probnum.randvars import Normal
    >>> from probnum.problems.zoo.diffeq import threebody_jax, vanderpol_jax
    >>> from probnum.randprocs.markov.integrator import IntegratedWienerProcess

    Compute the initial values of the restricted three-body problem as follows

    >>> ivp = threebody_jax()
    >>> print(ivp.y0)
    [ 0.994       0.          0.         -2.00158511]

    Construct the prior process.

    >>> prior_process = IntegratedWienerProcess(
    ...     initarg=ivp.t0, wiener_process_dimension=4, num_derivatives=3
    ... )

    Initialize with Taylor-mode autodiff.

    >>> taylor_init = TaylorMode()
    >>> improved_initrv = taylor_init(ivp=ivp, prior_process=prior_process)

    Print the results.

    >>> print(prior_process.transition.proj2coord(0) @ improved_initrv.mean)
    [ 0.994       0.          0.         -2.00158511]
    >>> print(improved_initrv.mean)
    [ 9.94000000e-01  0.00000000e+00 -3.15543023e+02  0.00000000e+00
      0.00000000e+00 -2.00158511e+00  0.00000000e+00  9.99720945e+04
      0.00000000e+00 -3.15543023e+02  0.00000000e+00  6.39028111e+07
     -2.00158511e+00  0.00000000e+00  9.99720945e+04  0.00000000e+00]

    Compute the initial values of the van-der-Pol oscillator as follows.
    First, set up the IVP and prior process.

    >>> ivp = vanderpol_jax()
    >>> print(ivp.y0)
    [2. 0.]
    >>> prior_process = IntegratedWienerProcess(
    ...     initarg=ivp.t0, wiener_process_dimension=2, num_derivatives=3
    ... )

    >>> taylor_init = TaylorMode()
    >>> improved_initrv = taylor_init(ivp=ivp, prior_process=prior_process)

    Print the results.

    >>> print(prior_process.transition.proj2coord(0) @ improved_initrv.mean)
    [2. 0.]
    >>> print(improved_initrv.mean)
    [    2.     0.    -2.    60.     0.    -2.    60. -1798.]
    >>> print(improved_initrv.std)
    [0. 0. 0. 0. 0. 0. 0. 0.]
    """

    def _compute_ode_derivatives(self, *, f, y0, num_derivatives):

        # Corner case 1: num_derivatives == 0
        derivs = [y0[:-1]]
        if num_derivatives == 0:
            return jnp.asarray(derivs)

        # Corner case 2: num_derivatives == 1
        y0_series = (jnp.ones_like(y0),)
        (y0_coeffs_taylor, [*y0_coeffs_remaining]) = jet(
            fun=f,
            primals=(y0,),
            series=(y0_series,),
        )
        derivs.append(y0_coeffs_taylor[:-1])
        if num_derivatives == 1:
            return jnp.asarray(derivs)

        # Order > 1
        for _ in range(1, num_derivatives):
            coeffs_taylor = (y0_coeffs_taylor, *y0_coeffs_remaining)
            (_, [*y0_coeffs_remaining]) = jet(
                fun=f,
                primals=(y0,),
                series=(coeffs_taylor,),
            )
            derivs.append(y0_coeffs_remaining[-2][:-1])

        return jnp.asarray(derivs)
