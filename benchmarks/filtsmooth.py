"""Benchmarks for Gaussian filtering."""
import functools

import numpy as np

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, randprocs, randvars


class Filtering:
    """Benchmark Kalman filter for different linearization techniques."""

    param_names = ["linearization_implementation"]
    params = [[("ekf", "classic"), ("ekf", "sqrt"), ("ukf", "classic")]]

    def setup(self, linearization_implementation):
        measvar = 0.1024
        initrv = randvars.Normal(np.ones(2), measvar * np.eye(2))
        rng = np.random.default_rng(seed=1)
        regression_problem, info = filtsmooth_zoo.pendulum(
            rng=rng,
            measurement_variance=measvar,
            timespan=(0.0, 4.0),
            step=0.0075,
            initrv=initrv,
        )
        prior_process = info["prior_process"]

        linearization, implementation = linearization_implementation
        _lin_method = {
            "ekf": functools.partial(
                filtsmooth.gaussian.approx.DiscreteEKFComponent,
                forward_implementation=implementation,
                backward_implementation=implementation,
            ),
            "ukf": filtsmooth.gaussian.approx.DiscreteUKFComponent,
        }[linearization]

        linearized_dynmod = _lin_method(prior_process.transition)
        linearized_measmod = _lin_method(regression_problem.measurement_models[0])
        regression_problem.measurement_models = [linearized_measmod] * len(
            regression_problem.locations
        )

        prior_process = randprocs.markov.MarkovProcess(
            transition=linearized_dynmod,
            initrv=prior_process.initrv,
            initarg=regression_problem.locations[0],
        )
        self.regression_problem = regression_problem

        self.kalman_filter = filtsmooth.gaussian.Kalman(prior_process=prior_process)

    def time_filter(self, linearization_implementation):
        self.kalman_filter.filter(self.regression_problem)

    def peakmem_filter(self, linearization_implementation):
        self.kalman_filter.filter(self.regression_problem)


class Smoothing:
    """Benchmark Kalman smoother for different linearization techniques."""

    param_names = ["linearization_implementation"]
    params = [[("ekf", "classic"), ("ekf", "sqrt"), ("ukf", "classic")]]

    def setup(self, linearization_implementation):
        measvar = 0.1024
        initrv = randvars.Normal(np.ones(2), measvar * np.eye(2))
        rng = np.random.default_rng(seed=1)
        regression_problem, info = filtsmooth_zoo.pendulum(
            rng=rng,
            measurement_variance=measvar,
            timespan=(0.0, 4.0),
            step=0.0075,
            initrv=initrv,
        )
        prior_process = info["prior_process"]

        linearization, implementation = linearization_implementation
        _lin_method = {
            "ekf": functools.partial(
                filtsmooth.gaussian.approx.DiscreteEKFComponent,
                forward_implementation=implementation,
                backward_implementation=implementation,
            ),
            "ukf": filtsmooth.gaussian.approx.DiscreteUKFComponent,
        }[linearization]

        linearized_dynmod = _lin_method(prior_process.transition)
        linearized_measmod = _lin_method(regression_problem.measurement_models[0])
        regression_problem.measurement_models = [linearized_measmod] * len(
            regression_problem.locations
        )

        prior_process = randprocs.markov.MarkovProcess(
            transition=linearized_dynmod,
            initrv=prior_process.initrv,
            initarg=regression_problem.locations[0],
        )

        self.kalman_filter = filtsmooth.gaussian.Kalman(prior_process=prior_process)
        self.filtering_posterior, _ = self.kalman_filter.filter(regression_problem)

    def time_smooth(self, linearization_implementation):
        self.kalman_filter.smooth(filter_posterior=self.filtering_posterior)

    def peakmem_smooth(self, linearization_implementation):
        self.kalman_filter.smooth(filter_posterior=self.filtering_posterior)


class DenseGridOperations:
    """Benchmark operations on a dense grid given the posteriors.

    That includes
        * Extrapolating / interpolating using the filter posterior
        * Extrapolating / interpolating using the smoothing posterior
        * Drawing samples from the smoothing posterior
    """

    param_names = ["linearization_implementation", "num_samples"]
    params = [[("ekf", "classic"), ("ekf", "sqrt"), ("ukf", "classic")], [1, 10]]

    def setup(self, linearization_implementation, num_samples):
        measvar = 0.1024
        initrv = randvars.Normal(np.ones(2), measvar * np.eye(2))
        rng = np.random.default_rng(seed=1)
        regression_problem, info = filtsmooth_zoo.pendulum(
            rng=rng,
            measurement_variance=measvar,
            timespan=(0.0, 4.0),
            step=0.0075,
            initrv=initrv,
        )
        prior_process = info["prior_process"]

        linearization, implementation = linearization_implementation
        _lin_method = {
            "ekf": functools.partial(
                filtsmooth.gaussian.approx.DiscreteEKFComponent,
                forward_implementation=implementation,
                backward_implementation=implementation,
            ),
            "ukf": filtsmooth.gaussian.approx.DiscreteUKFComponent,
        }[linearization]

        self.dense_locations = np.sort(
            np.unique(
                np.random.uniform(
                    low=regression_problem.locations[0],
                    high=1.2 * regression_problem.locations[-1],
                    size=int(1.2 * len(regression_problem.locations)),
                )
            )
        )

        linearized_dynmod = _lin_method(prior_process.transition)
        linearized_measmod = _lin_method(regression_problem.measurement_models[0])
        regression_problem.measurement_models = [linearized_measmod] * len(
            regression_problem.locations
        )

        prior_process = randprocs.markov.MarkovProcess(
            transition=linearized_dynmod,
            initrv=prior_process.initrv,
            initarg=regression_problem.locations[0],
        )

        self.kalman_filter = filtsmooth.gaussian.Kalman(prior_process=prior_process)

        self.filtering_posterior, _ = self.kalman_filter.filter(regression_problem)
        self.smoothing_posterior = self.kalman_filter.smooth(
            filter_posterior=self.filtering_posterior
        )

    def time_sample(self, linearization_implementation, num_samples):
        rng = np.random.default_rng(seed=1)
        self.smoothing_posterior.sample(
            rng=rng, t=self.dense_locations, size=num_samples
        )

    def peakmem_sample(self, linearization_implementation, num_samples):
        rng = np.random.default_rng(seed=1)
        self.smoothing_posterior.sample(
            rng=rng, t=self.dense_locations, size=num_samples
        )

    def time_dense_filter(self, linearization_implementation, num_samples):
        self.filtering_posterior(self.dense_locations)

    def peakmem_dense_filter(self, linearization_implementation, num_samples):
        self.filtering_posterior(self.dense_locations)

    def time_dense_smoother(self, linearization_implementation, num_samples):
        self.smoothing_posterior(self.dense_locations)

    def peakmem_dense_smoother(self, linearization_implementation, num_samples):
        self.smoothing_posterior(self.dense_locations)
