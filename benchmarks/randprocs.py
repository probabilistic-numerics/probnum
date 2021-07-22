"""Benchmarks for random processes."""
import numpy as np
import scipy.stats

from probnum import config, linops, randprocs, randvars, statespace


class MarkovProcessSampling:
    """Benchmark sampling from Markov processes."""

    param_names = ["lazy_linalg", "num_samples", "len_trajectory", "order", "dimension"]
    params = [[True, False], [10], [10], [5], [50, 100]]

    def setup(self, lazy_linalg, num_samples, len_trajectory, order, dimension):
        with config(lazy_linalg=lazy_linalg):
            dynamics = statespace.IBM(
                ordint=order,
                spatialdim=dimension,
                forward_implementation="classic",
                backward_implementation="classic",
            )

            measvar = 0.1024
            initrv = randvars.Normal(
                np.ones(dynamics.dimension),
                measvar * linops.Identity(dynamics.dimension),
            )

            time_domain = (0.0, float(len_trajectory))
            self.time_grid = np.arange(*time_domain)
            self.markov_process = randprocs.MarkovProcess(
                initarg=time_domain[0], initrv=initrv, transition=dynamics
            )

            rng = np.random.default_rng(seed=1)
            self.base_measure_realizations = scipy.stats.norm.rvs(
                size=((num_samples,) + self.time_grid.shape + initrv.shape),
                random_state=rng,
            )

    def time_sample(self, lazy_linalg, num_samples, len_trajectory, order, dimension):
        with config(lazy_linalg=lazy_linalg):
            for base_measure_real in self.base_measure_realizations:
                self.markov_process.transition.jointly_transform_base_measure_realization_list_forward(
                    base_measure_realizations=base_measure_real,
                    t=self.time_grid,
                    initrv=self.markov_process.initrv,
                    _diffusion_list=np.ones_like(self.time_grid[:-1]),
                )
