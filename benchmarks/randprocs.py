"""Benchmarks for random processes."""
import numpy as np
import scipy.stats

from probnum import config, linops, randprocs, randvars


class MarkovProcessSampling:
    """Benchmark sampling from Markov processes."""

    param_names = ["lazy_linalg", "len_trajectory", "num_derivatives", "dimension"]
    params = [[True, False], [10], [5], [50, 100]]

    def setup(self, lazy_linalg, len_trajectory, num_derivatives, dimension):
        with config(lazy_linalg=lazy_linalg):

            dynamics = randprocs.markov.integrator.IntegratedWienerTransition(
                num_derivatives=num_derivatives,
                wiener_process_dimension=dimension,
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
            self.markov_process = randprocs.markov.MarkovProcess(
                initarg=time_domain[0], initrv=initrv, transition=dynamics
            )

            rng = np.random.default_rng(seed=1)
            self.base_measure_realization = scipy.stats.norm.rvs(
                size=(self.time_grid.shape + initrv.shape),
                random_state=rng,
            )

    def time_sample(self, lazy_linalg, len_trajectory, num_derivatives, dimension):
        with config(lazy_linalg=lazy_linalg):
            self.markov_process.transition.jointly_transform_base_measure_realization_list_forward(
                base_measure_realizations=self.base_measure_realization,
                t=self.time_grid,
                initrv=self.markov_process.initrv,
                _diffusion_list=np.ones_like(self.time_grid[:-1]),
            )
