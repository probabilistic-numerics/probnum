"""Taylor-mode initialization."""
# pylint: disable=import-outside-toplevel


import numpy as np

from probnum import problems, randprocs, randvars, statespace
from probnum.diffeq.odefiltsmooth.initialize import _initialize


class TaylorModeInitialization(_initialize.InitializationRoutine):
    """Initialize a probabilistic ODE solver with Taylor-mode automatic differentiation.

    This requires JAX. For an explanation of what happens ``under the hood``, see [1]_.

    The implementation is inspired by the implementation in
    https://github.com/jacobjinkelly/easy-neural-ode/blob/master/latent_ode.py
    See also [2]_.

    References
    ----------
    .. [1] Krämer, N. and Hennig, P., Stable implementation of probabilistic ODE solvers,
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
    >>> from probnum.statespace import IBM
    >>> from probnum.randprocs import MarkovProcess

    Compute the initial values of the restricted three-body problem as follows

    >>> ivp = threebody_jax()
    >>> print(ivp.y0)
    [ 0.994       0.          0.         -2.00158511]

    Construct the prior process.

    >>> prior = IBM(ordint=3, spatialdim=4)
    >>> initrv = Normal(mean=np.zeros(prior.dimension), cov=np.eye(prior.dimension))
    >>> prior_process = MarkovProcess(transition=prior, initrv=initrv, initarg=ivp.t0)

    Initialize with Taylor-mode autodiff.

    >>> taylor_init = TaylorModeInitialization()
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
    >>> prior = IBM(ordint=3, spatialdim=2)
    >>> initrv = Normal(mean=np.zeros(prior.dimension), cov=np.eye(prior.dimension))
    >>> prior_process = MarkovProcess(transition=prior, initrv=initrv, initarg=ivp.t0)

    >>> taylor_init = TaylorModeInitialization()
    >>> improved_initrv = taylor_init(ivp=ivp, prior_process=prior_process)

    Print the results.

    >>> print(prior_process.transition.proj2coord(0) @ improved_initrv.mean)
    [2. 0.]
    >>> print(improved_initrv.mean)
    [    2.     0.    -2.    60.     0.    -2.    60. -1798.]
    >>> print(improved_initrv.std)
    [0. 0. 0. 0. 0. 0. 0. 0.]
    """

    def __init__(self):
        super().__init__(is_exact=True, requires_jax=True)

    def __call__(
        self, ivp: problems.InitialValueProblem, prior_process: randprocs.MarkovProcess
    ) -> randvars.RandomVariable:

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

        order = prior_process.transition.ordint

        dt = jnp.array([1.0])

        def evaluate_ode_for_extended_state(extended_state, ivp=ivp, dt=dt):
            r"""Evaluate the ODE for an extended state (x(t), t).

            More precisely, compute the derivative of the stacked state (x(t), t) according to the ODE.
            This function implements a rewriting of non-autonomous as autonomous ODEs.
            This means that

            .. math:: \dot x(t) = f(t, x(t))

            becomes

            .. math:: \dot z(t) = \dot (x(t), t) = (f(x(t), t), 1).

            Only considering autonomous ODEs makes the jet-implementation
            (and automatic differentiation in general) easier.
            """
            x, t = jnp.reshape(extended_state[:-1], ivp.y0.shape), extended_state[-1]
            dx = ivp.f(t, x)
            dx_ravelled = jnp.ravel(dx)
            stacked_ode_eval = jnp.concatenate((dx_ravelled, dt))
            return stacked_ode_eval

        def derivs_to_normal_randvar(derivs, ordint):
            """Finalize the output in terms of creating a suitably sized random
            variable."""
            all_derivs = statespace.Integrator._convert_derivwise_to_coordwise(
                np.asarray(derivs), ordint=ordint, spatialdim=ivp.y0.shape[0]
            )

            # Wrap all inputs through np.asarray, because 'Normal's
            # do not like JAX 'DeviceArray's
            return randvars.Normal(
                mean=np.asarray(all_derivs),
                cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
                cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
            )

        extended_state = jnp.concatenate((jnp.ravel(ivp.y0), jnp.array([ivp.t0])))
        derivs = []

        # Corner case 1: order == 0
        derivs.extend(ivp.y0)
        if order == 0:
            return derivs_to_normal_randvar(derivs=derivs, ordint=0)

        # Corner case 2: order == 1
        initial_series = (jnp.ones_like(extended_state),)
        (initial_taylor_coefficient, [*remaining_taylor_coefficents]) = jet(
            fun=evaluate_ode_for_extended_state,
            primals=(extended_state,),
            series=(initial_series,),
        )
        derivs.extend(initial_taylor_coefficient[:-1])
        if order == 1:
            return derivs_to_normal_randvar(derivs=derivs, ordint=1)

        # Order > 1
        for _ in range(1, order):
            new_series = (initial_taylor_coefficient, *remaining_taylor_coefficents)
            (_, [*remaining_taylor_coefficents]) = jet(
                fun=evaluate_ode_for_extended_state,
                primals=(extended_state,),
                series=(new_series,),
            )
            derivs.extend(remaining_taylor_coefficents[-2][:-1])
        return derivs_to_normal_randvar(derivs=derivs, ordint=order)
