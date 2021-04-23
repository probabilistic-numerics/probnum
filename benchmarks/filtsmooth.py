"""Benchmarks for Gaussian filtering."""
import functools

import numpy as np

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth


class Filtering:
    """Benchmark Kalman filter for different linearization techniques."""

    param_names = ["linearization_implementation"]
    params = [[("ekf", "classic"), ("ekf", "sqrt"), ("ukf", "classic")]]

    def setup(self, linearization_implementation):
        dynmod, measmod, initrv, regression_problem = filtsmooth_zoo.pendulum()
        linearization, implementation = linearization_implementation
        _lin_method = {
            "ekf": functools.partial(
                filtsmooth.DiscreteEKFComponent,
                forward_implementation=implementation,
                backward_implementation=implementation,
            ),
            "ukf": filtsmooth.DiscreteUKFComponent,
        }[linearization]

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.regression_problem = regression_problem
        self.kalman_filter = filtsmooth.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )

    def time_filter(self, linearization_implementation):
        self.kalman_filter.filter(self.regression_problem)

    def peakmem_filter(self, linearization_implementation):
        self.kalman_filter.filter(self.regression_problem)


class Smoothing:
    """Benchmark Kalman smoother for different linearization techniques."""

    param_names = ["linearization_implementation"]
    params = [[("ekf", "classic"), ("ekf", "sqrt"), ("ukf", "classic")]]

    def setup(self, linearization_implementation):
        dynmod, measmod, initrv, regression_problem = filtsmooth_zoo.pendulum()
        linearization, implementation = linearization_implementation
        _lin_method = {
            "ekf": functools.partial(
                filtsmooth.DiscreteEKFComponent,
                forward_implementation=implementation,
                backward_implementation=implementation,
            ),
            "ukf": filtsmooth.DiscreteUKFComponent,
        }[linearization]

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = filtsmooth.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )
        self.filtering_posterior = self.kalman_filter.filter(regression_problem)

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
        dynmod, measmod, initrv, regression_problem = filtsmooth_zoo.pendulum()
        linearization, implementation = linearization_implementation
        _lin_method = {
            "ekf": functools.partial(
                filtsmooth.DiscreteEKFComponent,
                forward_implementation=implementation,
                backward_implementation=implementation,
            ),
            "ukf": filtsmooth.DiscreteUKFComponent,
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

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = filtsmooth.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )
        self.filtering_posterior = self.kalman_filter.filter(regression_problem)
        self.smoothing_posterior = self.kalman_filter.smooth(
            filter_posterior=self.filtering_posterior
        )

    def time_sample(self, linearization_implementation, num_samples):
        self.smoothing_posterior.sample(t=self.dense_locations, size=num_samples)

    def peakmem_sample(self, linearization_implementation, num_samples):
        self.smoothing_posterior.sample(t=self.dense_locations, size=num_samples)

    def time_dense_filter(self, linearization_implementation, num_samples):
        self.filtering_posterior(self.dense_locations)

    def peakmem_dense_filter(self, linearization_implementation, num_samples):
        self.filtering_posterior(self.dense_locations)

    def time_dense_smoother(self, linearization_implementation, num_samples):
        self.smoothing_posterior(self.dense_locations)

    def peakmem_dense_smoother(self, linearization_implementation, num_samples):
        self.smoothing_posterior(self.dense_locations)
