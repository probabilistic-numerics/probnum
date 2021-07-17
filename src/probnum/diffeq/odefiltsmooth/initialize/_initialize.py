"""Interface for ODE filter initialization."""

import abc

import numpy as np
import scipy.integrate as sci

from probnum import filtsmooth, problems, statespace


class InitializationRoutine(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ivp, prior_process):
        pass


class RungeKuttaInitialization(InitializationRoutine):
    def __init__(self, dt=1e-2, method="RK45"):
        self.dt = dt
        self.method = method

    def __call__(self, ivp, prior_process):

        f, y0, t0, df = ivp.f, ivp.y0, ivp.t0, ivp.df
        y0 = np.asarray(y0)
        ode_dim = y0.shape[0] if y0.ndim > 0 else 1
        order = prior_process.transition.ordint

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
        measmod_scipy = statespace.DiscreteLTIGaussian(
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
        measmod_initcond = statespace.DiscreteLTIGaussian(
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
