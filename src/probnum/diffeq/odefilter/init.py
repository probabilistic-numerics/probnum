"""Initialization routines."""


import abc
from functools import partial
from typing import Optional

import numpy as np
import scipy.integrate as sci

from probnum import filtsmooth, problems, randprocs, randvars
from probnum.typing import FloatLike


class _InitializationRoutineBase(abc.ABC):
    """Interface for initialization routines for a filtering-based ODE solver.

    One crucial factor for stable implementation of probabilistic ODE solvers is
    starting with a good approximation of the derivatives of the initial condition [1]_.
    (This is common in all Nordsieck-like ODE solvers.)
    For this reason, efficient methods of initialization need to be devised.
    All initialization routines in ProbNum implement the interface :class:`InitializationRoutine`.

    References
    ----------
    .. [1] Krämer, N. and Hennig, P., Stable implementation of probabilistic ODE solvers,
       *arXiv:2012.10106*, 2020.
    """

    def __init__(self, is_exact: bool, requires_jax: bool):
        self._is_exact = is_exact
        self._requires_jax = requires_jax

    @abc.abstractmethod
    def __call__(
        self,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:
        raise NotImplementedError

    @property
    def is_exact(self) -> bool:
        """Exactness of the computed initial values.

        Some initialization routines yield the exact initial derivatives, some others
        only yield approximations.
        """
        return self._is_exact

    @property
    def requires_jax(self) -> bool:
        """Whether the implementation of the routine relies on JAX."""
        return self._requires_jax


class TaylorMode(_InitializationRoutineBase):
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
    >>> from probnum.randprocs.markov.integrator import IntegratedWienerProcess

    Compute the initial values of the restricted three-body problem as follows

    >>> ivp = threebody_jax()
    >>> print(ivp.y0)
    [ 0.994       0.          0.         -2.00158511]

    Construct the prior process.

    >>> prior_process = IntegratedWienerProcess(initarg=ivp.t0, wiener_process_dimension=4, num_derivatives=3)

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
    >>> prior_process = IntegratedWienerProcess(initarg=ivp.t0, wiener_process_dimension=2, num_derivatives=3)

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

    def __init__(self):
        super().__init__(is_exact=True, requires_jax=True)

    def __call__(
        self,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:

        _, jnp, _, jet = _import_jax()
        num_derivatives = prior_process.transition.num_derivatives
        dt = jnp.array([1.0])
        evaluate_ode_as_autonomous_ode = partial(
            _evaluate_ode_as_autonomous_ode, ivp=ivp, dt=dt, jnp=jnp
        )
        derivs_to_normal_randvar = partial(_derivs_to_normal_randvar, ivp=ivp, jnp=jnp)

        extended_state = jnp.concatenate((jnp.ravel(ivp.y0), jnp.array([ivp.t0])))
        derivs = []

        # Corner case 1: num_derivatives == 0
        derivs.extend(ivp.y0)
        if num_derivatives == 0:
            return derivs_to_normal_randvar(
                derivs=derivs, num_derivatives_in_prior=num_derivatives
            )

        # Corner case 2: num_derivatives == 1
        initial_series = (jnp.ones_like(extended_state),)
        (initial_taylor_coefficient, [*remaining_taylor_coefficents]) = jet(
            fun=evaluate_ode_as_autonomous_ode,
            primals=(extended_state,),
            series=(initial_series,),
        )
        derivs.extend(initial_taylor_coefficient[:-1])
        if num_derivatives == 1:
            return derivs_to_normal_randvar(
                derivs=derivs, num_derivatives_in_prior=num_derivatives
            )

        # Order > 1
        for _ in range(1, num_derivatives):
            taylor_coefficients = (
                initial_taylor_coefficient,
                *remaining_taylor_coefficents,
            )
            (_, [*remaining_taylor_coefficents]) = jet(
                fun=evaluate_ode_as_autonomous_ode,
                primals=(extended_state,),
                series=(taylor_coefficients,),
            )
            derivs.extend(remaining_taylor_coefficents[-2][:-1])
        return derivs_to_normal_randvar(
            derivs=derivs, num_derivatives_in_prior=num_derivatives
        )


def _evaluate_ode_as_autonomous_ode(extended_state, ivp, dt, jnp):
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


def _derivs_to_normal_randvar(derivs, num_derivatives_in_prior, ivp, jnp):
    """Finalize the output in terms of creating a suitably sized random variable."""
    all_derivs = randprocs.markov.integrator.convert.convert_derivwise_to_coordwise(
        np.asarray(derivs),
        num_derivatives=num_derivatives_in_prior,
        wiener_process_dimension=ivp.y0.shape[0],
    )

    # Wrap all inputs through np.asarray, because 'Normal's
    # do not like JAX 'DeviceArray's
    return randvars.Normal(
        mean=np.asarray(all_derivs),
        cov=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
        cov_cholesky=np.asarray(jnp.diag(jnp.zeros(len(derivs)))),
    )


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
        from jax.config import config
        from jax.experimental.jet import jet

        config.update("jax_enable_x64", True)

        return jax, jnp, config, jet

    except ImportError as err:
        raise ImportError(
            "Cannot perform Jax-based initialization without the optional "
            "dependencies jax and jaxlib. Try installing them via `pip install jax jaxlib`."
        ) from err


class _AutoDiffBase(_InitializationRoutineBase):
    def __init__(self):
        super().__init__(is_exact=True, requires_jax=True)

    def __call__(
        self,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:

        dim = ivp.dimension + 1
        num_derivatives = prior_process.transition.num_derivatives

        jax, jnp, _, _ = _import_jax()
        mean_matrix = self._compute_ode_derivatives(ivp, num_derivatives, jax, jnp)
        mean = mean_matrix.reshape((-1,), order="F")
        cov = jnp.zeros((mean.shape[0], mean.shape[0]))
        return randvars.Normal(mean=np.asarray(mean), cov=np.asarray(cov))

    def _compute_ode_derivatives(self, ivp, num_derivatives, jax, jnp):
        f, y0 = self._make_autonomous(ivp=ivp, jnp=jnp)
        gen = self._F_generator(f=f, y0=y0, jax=jax)
        mean_matrix = jnp.stack(
            [next(gen)(y0)[:-1] for _ in range(num_derivatives + 1)]
        )
        return mean_matrix

    def _make_autonomous(self, *, ivp, jnp):
        y0_autonomous = jnp.concatenate([ivp.y0, jnp.array([ivp.t0])])

        def f_autonomous(y, /):
            x, t = y[:-1], y[-1]
            fx = ivp.f(t, x)
            return jnp.concatenate([fx, jnp.array([1.0])])

        return f_autonomous, y0_autonomous

    def _F_generator(self, f, y0, jax):
        def fwd_deriv(f, f0):
            def ff(x):
                return f(x)

            def df(x):
                return self.jvp_or_vjp(fun=f, primals=x, tangents=f0(x), jax=jax)

            return df

        yield lambda x: y0

        g = f
        f0 = f
        while True:
            yield g
            g = fwd_deriv(g, f0)

    def jvp_or_vjp(self, *, fun, primals, tangents, jax):
        raise NotImplementedError


class ForwardMode(_AutoDiffBase):
    def jvp_or_vjp(self, *, fun, primals, tangents, jax):
        _, y = jax.jvp(fun, (primals,), (tangents,))
        return y


class ForwardModeNaive(_AutoDiffBase):
    def jvp_or_vjp(self, *, fun, primals, tangents, jax):
        return jax.jacfwd(fun)(primals) @ tangents


class ReverseModeNaive(_AutoDiffBase):
    def jvp_or_vjp(self, *, fun, primals, tangents, jax):
        # The following should work, but doesn't
        # y, dfx_fun = jax.vjp(fun, primals)
        # a, = dfx_fun(tangents)
        # return a

        # Therefore we go sledge-hammer
        return jax.jacrev(fun)(primals) @ tangents


class RungeKutta(_InitializationRoutineBase):
    r"""Initialize a probabilistic ODE solver by fitting the prior process to a few steps of an approximate ODE solution computed with Scipy's Runge-Kutta methods.

    Parameters
    ----------
    dt
        Maximum step-size to use for computing the approximate ODE solution. The smaller, the more accurate, but also, the smaller, the less stable.
        The best value here depends on the ODE problem, and probably the chosen method. Optional. Default is ``1e-2``.
    method
        Which solver to use. This is communicated as a string that is compatible with ``scipy.integrate.solve_ivp(..., method=method)``.
        Optional. Default is `DOP853`.

    Examples
    --------

    >>> import numpy as np
    >>> from probnum.randvars import Normal
    >>> from probnum.problems.zoo.diffeq import vanderpol
    >>> from probnum.randprocs.markov.integrator import IntegratedWienerProcess

    Compute the initial values of the van-der-Pol problem as follows.
    First, we set up the ODE problem and the prior process.

    >>> ivp = vanderpol()
    >>> print(ivp.y0)
    [2. 0.]
    >>> prior_process = IntegratedWienerProcess(initarg=ivp.t0, num_derivatives=3, wiener_process_dimension=2)

    Next, we call the initialization routine.

    >>> rk_init = RungeKuttaInitialization()
    >>> improved_initrv = rk_init(ivp=ivp, prior_process=prior_process)
    >>> print(prior_process.transition.proj2coord(0) @ improved_initrv.mean)
    [2. 0.]
    >>> print(np.round(improved_initrv.mean, 1))
    [    2.      0.     -2.     58.2     0.     -2.     60.  -1745.7]
    >>> print(np.round(np.log10(improved_initrv.std), 1))
    [-13.8 -11.3  -9.   -1.5 -13.8 -11.3  -9.   -1.5]
    """

    def __init__(
        self, dt: Optional[FloatLike] = 1e-2, method: Optional[str] = "DOP853"
    ):
        self.dt = dt
        self.method = method
        super().__init__(is_exact=False, requires_jax=False)

    def __call__(
        self,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:
        """Compute the initial distribution.

        For Runge-Kutta initialization, it goes as follows:

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
        ivp
            Initial value problem.
        prior_process
            Prior Gauss-Markov process.

        Returns
        -------
        Normal
            Estimated (improved) initial random variable. Compatible with the specified prior.
        """
        f, y0, t0, df = ivp.f, ivp.y0, ivp.t0, ivp.df
        y0 = np.asarray(y0)
        ode_dim = y0.shape[0] if y0.ndim > 0 else 1
        order = prior_process.transition.num_derivatives

        # order + 1 would suffice in theory, 2*order + 1 is for good measure
        # (the "+1" is a safety factor for order=1)
        num_steps = 2 * order + 1
        t_eval = np.arange(t0, t0 + (num_steps + 1) * self.dt, self.dt)
        sol = sci.solve_ivp(
            f,
            (t0, t0 + (num_steps + 1) * self.dt),
            y0=y0,
            atol=1e-12,
            rtol=1e-12,
            t_eval=t_eval,
            method=self.method,
        )

        # Measurement model for SciPy observations
        proj_to_y = prior_process.transition.proj2coord(coord=0)
        zeros_shift = np.zeros(ode_dim)
        zeros_cov = np.zeros((ode_dim, ode_dim))
        measmod_scipy = randprocs.markov.discrete.LTIGaussian(
            proj_to_y,
            zeros_shift,
            zeros_cov,
            proc_noise_cov_cholesky=zeros_cov,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )

        # Measurement model for initial condition observations
        proj_to_dy = prior_process.transition.proj2coord(coord=1)
        if df is not None and order > 1:
            proj_to_ddy = prior_process.transition.proj2coord(coord=2)
            projmat_initial_conditions = np.vstack((proj_to_y, proj_to_dy, proj_to_ddy))
            initial_data = np.hstack((y0, f(t0, y0), df(t0, y0) @ f(t0, y0)))
        else:
            projmat_initial_conditions = np.vstack((proj_to_y, proj_to_dy))
            initial_data = np.hstack((y0, f(t0, y0)))
        zeros_shift = np.zeros(len(projmat_initial_conditions))
        zeros_cov = np.zeros(
            (len(projmat_initial_conditions), len(projmat_initial_conditions))
        )
        measmod_initcond = randprocs.markov.discrete.LTIGaussian(
            projmat_initial_conditions,
            zeros_shift,
            zeros_cov,
            proc_noise_cov_cholesky=zeros_cov,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )

        # Create regression problem and measurement model list
        ts = sol.t[:num_steps]
        ys = list(sol.y[:, :num_steps].T)
        ys[0] = initial_data
        measmod_list = [measmod_initcond] + [measmod_scipy] * (len(ts) - 1)
        regression_problem = problems.TimeSeriesRegressionProblem(
            observations=ys, locations=ts, measurement_models=measmod_list
        )

        # Infer the solution
        kalman = filtsmooth.gaussian.Kalman(prior_process)
        out, _ = kalman.filtsmooth(regression_problem)
        estimated_initrv = out.states[0]
        return estimated_initrv
