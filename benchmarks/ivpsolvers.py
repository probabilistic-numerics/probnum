"""Benchmarks for probabilistic IVP solvers."""
import numpy as np

from probnum.diffeq import lotkavolterra, probsolve_ivp
from probnum.random_variables import Constant


def load_lotkavolterra():
    """Load LV system as a basic IVP."""
    initrv = Constant(np.array([20, 20]))
    return lotkavolterra(
        timespan=[0, 0.55], initrv=initrv, params=(0.5, 0.05, 0.5, 0.05)
    )


class IVPSolve:
    """Benchmark ODE-filter and ODE-smoother with fixed steps and different priors and
    information operators."""

    param_names = ["method", "prior"]
    params = [["ek0", "ek1"], ["IBM3", "IOUP3", "MAT72"]]

    def setup(self, method, prior):
        # pylint: disable=invalid-name
        self.ivp = load_lotkavolterra()
        self.stepsize = 1e-2

    def time_solve(self, method, prior):
        f = self.ivp.f
        df = self.ivp.df
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean
        probsolve_ivp(
            f,
            t0,
            tmax,
            y0,
            df=df,
            method=method,
            which_prior=prior,
            step=self.stepsize,
            atol=None,
            rtol=None,
        )

    def peakmem_solve(self, method, prior):
        f = self.ivp.f
        df = self.ivp.df
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean
        probsolve_ivp(
            f,
            t0,
            tmax,
            y0,
            df=df,
            method=method,
            which_prior=prior,
            step=self.stepsize,
            atol=None,
            rtol=None,
        )
