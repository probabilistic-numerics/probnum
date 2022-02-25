"""Initialization routines based on fitting the prior to (a few steps of a) non-
probabilistc solver."""


from typing import Optional

import numpy as np
import scipy.integrate as sci

from probnum import filtsmooth, problems, randprocs, randvars
from probnum.typing import FloatLike

from ._interface import InitializationRoutine


class _NonProbabilisticFitBase(InitializationRoutine):
    """Fit the prior process to a few steps of a non-probabilistic solver."""

    def __init__(
        self, *, dt: FloatLike = 1e-2, observation_noise_std: FloatLike = 1e-7
    ):
        super().__init__(is_exact=False, requires_jax=False)
        self._dt = dt
        self._observation_noise_std = observation_noise_std

    def __call__(
        self,
        *,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:

        num_steps = prior_process.transition.num_derivatives + 1
        t_eval = np.linspace(
            start=ivp.t0,
            stop=ivp.t0 + num_steps * self._dt,
            num=num_steps,
            endpoint=True,
        )

        data = self._data(ivp=ivp, t_eval=t_eval)
        rv = self._improve(data=data, prior_process=prior_process)
        return rv

    def _data(self, *, ivp, t_eval):
        raise NotImplementedError

    def _improve(self, *, data, prior_process):

        # Measurement model for SciPy observations
        ode_dim = prior_process.transition.wiener_process_dimension
        proj_to_y = prior_process.transition.proj2coord(coord=0)
        observation_noise_std = self._observation_noise_std * np.ones(ode_dim)
        process_noise = randvars.Normal(
            mean=np.zeros(ode_dim),
            cov=np.diag(observation_noise_std**2),
            cov_cholesky=np.diag(observation_noise_std),
        )
        measmod_scipy = randprocs.markov.discrete.LTIGaussian(
            transition_matrix=proj_to_y,
            noise=process_noise,
            forward_implementation="sqrt",
            backward_implementation="sqrt",
        )

        # Regression problem
        ts, ys = data
        regression_problem = problems.TimeSeriesRegressionProblem(
            observations=ys, locations=ts, measurement_models=[measmod_scipy] * len(ts)
        )

        # Infer the solution
        kalman = filtsmooth.gaussian.Kalman(prior_process)
        out, _ = kalman.filtsmooth(regression_problem)
        estimated_initrv = out.states[0]
        return estimated_initrv


class NonProbabilisticFit(_NonProbabilisticFitBase):
    """Fit the prior process to a few steps of a non-probabilistic solver."""

    def __init__(
        self,
        *,
        dt: Optional[FloatLike] = 1e-2,
        observation_noise_std=1e-7,
        method: str = "DOP853",
    ):
        super().__init__(dt=dt, observation_noise_std=observation_noise_std)
        self._method = method

    def _data(self, *, ivp, t_eval):

        sol = sci.solve_ivp(
            fun=ivp.f,
            t_span=(np.amin(t_eval), np.amax(t_eval)),
            y0=ivp.y0,
            atol=1e-12,
            rtol=1e-12,
            t_eval=t_eval,
            method=self._method,
        )
        ts = sol.t
        ys = sol.y.T
        return ts, ys


class NonProbabilisticFitWithJacobian(_NonProbabilisticFitBase):
    """Fit the prior process to a few steps of a non-probabilistic solver and use
    Jacobians."""

    def __init__(
        self,
        *,
        dt: Optional[FloatLike] = 1e-2,
        observation_noise_std=1e-7,
        method: str = "Radau",
    ):
        super().__init__(dt=dt, observation_noise_std=observation_noise_std)
        self._method = method

    def _data(self, *, ivp, t_eval):

        sol = sci.solve_ivp(
            fun=ivp.f,
            jac=ivp.df,
            t_span=(np.amin(t_eval), np.amax(t_eval)),
            y0=ivp.y0,
            atol=1e-12,
            rtol=1e-12,
            t_eval=t_eval,
            method=self._method,
        )
        ts = sol.t
        ys = sol.y.T
        return ts, ys
