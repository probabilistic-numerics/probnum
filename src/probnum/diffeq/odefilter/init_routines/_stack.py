"""Stacking-based initialization routines."""


import numpy as np

from probnum import problems, randprocs, randvars

from ._interface import InitializationRoutine


class _StackBase(InitializationRoutine):
    def __init__(self, *, scale_cholesky=1e3):
        super().__init__(is_exact=False, requires_jax=False)
        self._scale_cholesky = scale_cholesky

    def __call__(
        self,
        *,
        ivp: problems.InitialValueProblem,
        prior_process: randprocs.markov.MarkovProcess,
    ) -> randvars.RandomVariable:

        mean_matrix, std_matrix = self._stack_initial_states(
            ivp=ivp, num_derivatives=prior_process.transition.num_derivatives
        )

        mean = mean_matrix.reshape((-1,), order="F")
        std = std_matrix.reshape((-1,), order="F")
        return randvars.Normal(
            mean=np.asarray(mean),
            cov=np.diag(std**2),
            cov_cholesky=np.diag(std),
        )

    def _stack_initial_states(self, *, ivp, num_derivatives):
        raise NotImplementedError


class Stack(_StackBase):
    """Initialization by stacking y0, f(y0)."""

    def _stack_initial_states(self, *, ivp, num_derivatives):
        d, n = ivp.y0.shape[0], num_derivatives + 1

        fy = ivp.f(ivp.t0, ivp.y0)

        mean = np.stack([ivp.y0, fy] + [np.zeros(d)] * (n - 2))
        std = np.stack(
            [0.0 * ivp.y0, 0.0 * fy] + [self._scale_cholesky * np.ones(d)] * (n - 2)
        )
        return mean, std


class StackWithJacobian(_StackBase):
    """Initialization by stacking y0, f(y0), and df(y0)."""

    def _stack_initial_states(self, *, ivp, num_derivatives):
        d, n = ivp.y0.shape[0], num_derivatives + 1

        fy = ivp.f(ivp.t0, ivp.y0)
        dfy = ivp.df(ivp.t0, ivp.y0) @ fy

        mean = np.stack([ivp.y0, fy, dfy] + [np.zeros(d)] * (n - 3))
        std = np.stack(
            [0.0 * ivp.y0, 0.0 * fy, 0.0 * dfy]
            + [self._scale_cholesky * np.ones(d)] * (n - 3)
        )
        return mean, std
