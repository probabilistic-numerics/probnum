"""Benchmarks for random processes."""
import numpy as np

from probnum import config, randprocs, randvars, statespace


class MarkovProcessSampling:
    """Benchmark sampling from Markov processes."""

    param_names = ["use_linops", "num_samples", "len_trajectory"]
    params = [[True, False], [100], [100]]

    def setup(self, use_linops, num_samples, len_trajectory):
        with config(statespace_use_linops=use_linops):
            dynamics = statespace.IBM(
                ordint=5,
                spatialdim=3,
                forward_implementation="classic",
                backward_implementation="classic",
            )

        measvar = 0.1024
        initrv = randvars.Normal(
            np.ones(dynamics.dimension), measvar * np.eye(dynamics.dimension)
        )

        time_domain = (0.0, float(len_trajectory))
        self.time_grid = np.arange(*time_domain)
        self.prior_process = randprocs.MarkovProcess(
            initarg=time_domain[0], initrv=initrv, transition=dynamics
        )
        self.rng = np.random.default_rng(seed=1)

    def time_sample(self, use_linops, num_samples, len_trajectory):
        self.prior_process.sample(self.rng, args=self.time_grid, size=num_samples)
