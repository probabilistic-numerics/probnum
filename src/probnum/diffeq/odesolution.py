"""ODESolution object, returned by `probsolve_ivp`

Contains the discrete time and function outputs.
Provides dense output by being callable.
Can function values can also be accessed by indexing.
"""
import numpy as np

from probnum.prob import RandomVariable, Normal


class ODESolution:
    """Continuous ODE Solution"""

    def __init__(self, times, rvs, solver):
        self.solver = solver
        self.d = self.solver.ivp.ndim

        self._t = times
        self._state_rvs = _RandomVariableList(rvs)

        self._y = [
            RandomVariable(
                distribution=Normal(
                    rv.mean()[0 :: self.d], rv.cov()[0 :: self.d, 0 :: self.d]
                )
            )
            for rv in rvs
        ]

    @property
    def t(self):
        return self._t

    @property
    def y(self):
        return self._y

    @property
    def state_rvs(self):
        """Return the posterior over states after undoing the preconditioning"""
        state_rvs = _RandomVariableList(
            [self.solver.undo_preconditioning_rv(rv) for rv in self._state_rvs]
        )
        return state_rvs

    def __call__(self, t, smoothed=True):
        """Evaluate the solution at time t

        Algorithm:
        1. Find closest t_prev and t_next, with t_prev < t < t_next
        2. Predict from t_prev to t
        3. Predict from t to t_next
        4. Smooth from t_next to t
        5. Return random variable for time t
        """

        d = self.solver.ivp.ndim
        q = self.solver.gfilt.dynamod.ordint

        if t in self.t:
            idx = (self.t <= t).sum() - 1
            out_rv = self._state_rvs[idx]
        else:
            prev_idx = (self.t < t).sum() - 1
            prev_time = self.t[prev_idx]
            prev_rv = self._state_rvs[prev_idx]

            predicted, _ = self.solver.gfilt.predict(
                start=prev_time, stop=t, randvar=prev_rv
            )
            out_rv = predicted

            if smoothed:
                next_time = self.t[prev_idx + 1]
                next_rv = self._state_rvs[prev_idx + 1]
                next_pred, crosscov = self.solver.gfilt.predict(
                    start=t, stop=next_time, randvar=predicted
                )

                smoothed = self.solver.gfilt.smooth_step(
                    predicted, next_pred, next_rv, crosscov
                )

                out_rv = smoothed

        out_rv = self.solver.undo_preconditioning_rv(out_rv)

        f_mean = out_rv.mean()[0::d]
        f_cov = out_rv.cov()[0::d, 0::d]

        return RandomVariable(distribution=Normal(f_mean, f_cov))

    def __len__(self):
        """Number of points in the discrete solution

        Note that the length of `self.t` and `self.y` should coincide.
        """
        return len(self.t)

    def __getitem__(self, idx):
        """Access the discrete solution through indexing

        Note that the stored `self._state_rvs` are still in the transformed,
        "preconditioned" space. Therefore we need to first undo the preconditioning
        before returning them.
        """
        if isinstance(idx, int):
            rv = self._state_rvs[idx]
            rv = self.solver.undo_preconditioning_rv(rv)
            f_mean = rv.mean()[0 :: self.d]
            f_cov = rv.cov()[0 :: self.d, 0 :: self.d]
            return RandomVariable(distribution=Normal(f_mean, f_cov))
        elif isinstance(idx, slice):
            rvs = self._state_rvs[idx]
            rvs = [self.solver.undo_preconditioning_rv(rv) for rv in rvs]
            f_means = [rv.mean()[0 :: self.d] for rv in rvs]
            f_covs = [rv.cov()[0 :: self.d, 0 :: self.d] for rv in rvs]
            f_rvs = [
                RandomVariable(distribution=Normal(f_mean, f_cov))
                for (f_mean, f_cov) in zip(f_means, f_covs)
            ]
            return f_rvs
        else:
            raise ValueError("Invalid index")


class _RandomVariableList(list):
    @property
    def mean(self):
        return np.stack([rv.mean() for rv in self])

    @property
    def cov(self):
        return np.stack([rv.cov() for rv in self])

    @property
    def var(self):
        return np.stack([rv.distribution.var() for rv in self])

    @property
    def std(self):
        return np.stack([rv.distribution.std() for rv in self])
