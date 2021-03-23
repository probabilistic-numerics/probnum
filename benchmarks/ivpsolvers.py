"""Benchmarks for probabilistic IVP solvers."""
import numpy as np

from probnum.diffeq import lotkavolterra, probsolve_ivp
from probnum.randvars import Constant


def load_lotkavolterra():
    """Load LV system as a basic IVP."""
    initrv = Constant(np.array([20, 20]))
    return lotkavolterra(
        timespan=[0, 0.55], initrv=initrv, params=(0.5, 0.05, 0.5, 0.05)
    )


class IVPSolve:
    """Benchmark ODE-filter and ODE-smoother with fixed steps and different priors and
    information operators."""

    param_names = ["method", "algo_order"]
    params = [["ek0", "ek1"], [2, 3]]

    def setup(self, method, prior):
        # pylint: disable=invalid-name
        self.ivp = load_lotkavolterra()
        self.stepsize = 1e-1

    def time_solve(self, method, algo_order):
        f = self.ivp.rhs
        df = self.ivp.jacobian
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean
        probsolve_ivp(
            f,
            t0,
            tmax,
            y0,
            df=df,
            method=method,
            algo_order=algo_order,
            step=self.stepsize,
            adaptive=False,
        )

    def peakmem_solve(self, method, algo_order):
        f = self.ivp.rhs
        df = self.ivp.jacobian
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean
        probsolve_ivp(
            f,
            t0,
            tmax,
            y0,
            df=df,
            method=method,
            algo_order=algo_order,
            step=self.stepsize,
            adaptive=False,
        )
