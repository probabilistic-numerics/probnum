"""Runge-Kutta initialisation."""


from typing import Optional

import numpy as np
import scipy.integrate as sci

from probnum import filtsmooth, problems, randprocs, randvars
from probnum.diffeq.odefiltsmooth.initialization_routines import _initialization_routine
from probnum.typing import FloatArgType


class RungeKuttaInitialization(_initialization_routine.InitializationRoutine):
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
        self, dt: Optional[FloatArgType] = 1e-2, method: Optional[str] = "DOP853"
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
        measmod_scipy = randprocs.markov.discrete.DiscreteLTIGaussian(
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
        measmod_initcond = randprocs.markov.discrete.DiscreteLTIGaussian(
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
