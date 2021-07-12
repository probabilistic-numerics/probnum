"""Benchmarks for probabilistic IVP solvers."""
import numpy as np

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum.diffeq import probsolve_ivp


def load_lotkavolterra():
    """Load LV system as a basic IVP."""
    y0 = np.array([20, 20])
    return diffeq_zoo.lotkavolterra(
        t0=0.0, tmax=0.55, y0=y0, params=(0.5, 0.05, 0.5, 0.05)
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
        probsolve_ivp(
            f=self.ivp.f,
            t0=self.ivp.t0,
            tmax=self.ivp.tmax,
            y0=self.ivp.y0,
            df=self.ivp.df,
            method=method,
            dense_output=True,
            algo_order=algo_order,
            step=self.stepsize,
            adaptive=False,
        )

    def peakmem_solve(self, method, algo_order):
        probsolve_ivp(
            f=self.ivp.f,
            t0=self.ivp.t0,
            tmax=self.ivp.tmax,
            y0=self.ivp.y0,
            df=self.ivp.df,
            method=method,
            dense_output=True,
            algo_order=algo_order,
            step=self.stepsize,
            adaptive=False,
        )
