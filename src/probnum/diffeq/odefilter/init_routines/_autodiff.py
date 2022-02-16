"""Automatic-differentiation-based initialization routines."""

import itertools

import numpy as np

from probnum import problems, randprocs, randvars

from ._interface import InitializationRoutine

# pylint: disable="import-outside-toplevel"
try:
    import jax
    from jax.config import config
    from jax.experimental.jet import jet
    import jax.numpy as jnp

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

    References
    ----------
    .. [1] Krämer, N. and Hennig, P.,
       Stable implementation of probabilistic ODE solvers,
       *arXiv:2012.10106*, 2020.


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
        taylor_coefficients = self._taylor_approximation(
            f=f, y0=y0, order=num_derivatives
        )
        return taylor_coefficients[:, :-1]

    def _taylor_approximation(self, *, f, y0, order):
        """Compute an `n`th order Taylor approximation of f at y0."""
        taylor_coefficient_gen = self._taylor_coefficient_generator(f=f, y0=y0)

        # access the nth entry via itertools.islice
        derivatives = next(itertools.islice(taylor_coefficient_gen, order - 1, None))

        derivatives_as_array = jnp.stack(derivatives, axis=0)
        return derivatives_as_array

    @staticmethod
    def _taylor_coefficient_generator(*, f, y0):
        """Generate Taylor coefficients of `f` via `jet`()."""

        # This is the position where we evaluate the
        # derivatives of the ODE solution.
        # It is the 0th Taylor coefficient.
        x_primals = y0

        # This contains the higher-order, unnormalised
        # Taylor coefficients of the input (i.e., the ODE solution).
        x_series = (f(y0),)

        while True:
            yield (x_primals,) + x_series

            # jet() computes a Taylor approximation of g(t) := f(x(t))
            # The output is the zeroth Taylor approximation g(t_0) ('primals')
            # as well its higher-order Taylor coefficients ('series')
            g_primals, g_series = jet(fun=f, primals=(x_primals,), series=(x_series,))

            # For ODEs \dot y(t) = f(y(t)),
            # The nth Taylor coefficient of y is the
            # (n-1)th Taylor coefficient of g(t) = f(y(t)).
            # This way, by augmenting x0 with the Taylor series
            # approximating g(t) = f(y(t)), we increase the order
            # of the approximation by 1.
            x_series = (g_primals, *g_series)
