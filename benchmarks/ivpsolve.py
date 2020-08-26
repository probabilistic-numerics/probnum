"""
Benchmarks for probabilistic ivp solvers
"""
import numpy as np
from probnum.diffeq import lotkavolterra, probsolve_ivp
from probnum.prob import RandomVariable, Dirac


def load_lotkavolterra():
    """Load LV system as a basic IVP."""
    initrv = RandomVariable(distribution=Dirac(np.array([20, 20])))
    return lotkavolterra(
        timespan=[0, 0.55], initrv=initrv, params=(0.5, 0.05, 0.5, 0.05)
    )


class IVPSolve:
    """Benchmark ODE-filter and ODE-smoother on small steps with high-order priors"""

    param_names = ["method", "precond", "prior"]
    params = [["eks0", "ekf0"], ["with", "without"], ["ibm4", "ioup4", "matern92"]]

    def setup(self, method, precond, prior):
        self.ivp = load_lotkavolterra()
        self.stepsize = 1e-2

    def time_solve(self, method, precond, prior):
        precond_step = self.stepsize if precond == "with" else 1.0
        probsolve_ivp(
            self.ivp,
            method=method,
            which_prior=prior,
            step=self.stepsize,
            precond_step=precond_step,
        )

    def peakmem_solve(self, method, precond, prior):
        precond_step = self.stepsize if precond == "with" else 1.0
        probsolve_ivp(
            self.ivp,
            method=method,
            which_prior=prior,
            step=self.stepsize,
            precond_step=precond_step,
        )
