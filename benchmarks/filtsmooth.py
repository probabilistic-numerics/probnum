"""Benchmarks for Gaussian filtering."""
import numpy as np

from probnum import filtsmooth, problems, randvars, statespace


def load_pendulum():
    """Set up pendulum problem"""

    np.random.seed(12345)

    delta_t = 0.0075
    var = 0.32 ** 2
    g = 9.81

    def f(t, x):
        x1, x2 = x
        y1 = x1 + x2 * delta_t
        y2 = x2 - g * np.sin(x1) * delta_t
        return np.array([y1, y2])

    def df(t, x):
        x1, x2 = x
        y1 = [1, delta_t]
        y2 = [-g * np.cos(x1) * delta_t, 1]
        return np.array([y1, y2])

    def h(t, x):
        x1, x2 = x
        return np.array([np.sin(x1)])

    def dh(t, x):
        x1, x2 = x
        return np.array([[np.cos(x1), 0.0]])

    q = 1.0 * (
        np.diag(np.array([delta_t ** 3 / 3, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2]), 1)
        + np.diag(np.array([delta_t ** 2 / 2]), -1)
    )
    r = var * np.eye(1)
    initmean = np.ones(2)
    initcov = var * np.eye(2)
    dynamod = statespace.DiscreteGaussian(2, 2, f, lambda t: q, df)
    measmod = statespace.DiscreteGaussian(2, 1, h, lambda t: r, dh)
    initrv = randvars.Normal(initmean, initcov)
    return dynamod, measmod, initrv, {"dt": delta_t, "tmax": 4}


class Filtering:
    """Benchmark Kalman filter for different linearization techniques."""

    param_names = ["linearization"]
    params = [["ekf", "ukf"]]

    def setup(self, linearization):
        dynmod, measmod, initrv, info = load_pendulum()
        _lin_method = {
            "ekf": filtsmooth.DiscreteEKFComponent,
            "ukf": filtsmooth.DiscreteUKFComponent,
        }[linearization]

        self.locations = np.arange(0.0, info["tmax"], step=info["dt"])
        _, self.observations = statespace.generate_samples(
            dynmod=dynmod, measmod=measmod, initrv=initrv, times=self.locations
        )
        self.regression_problem = problems.RegressionProblem(
            observations=self.observations, locations=self.locations
        )

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = filtsmooth.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )

    def time_filter(self, linearization):
        self.kalman_filter.filter(self.regression_problem)

    def peakmem_filter(self, linearization):
        self.kalman_filter.filter(self.regression_problem)


class Smoothing:
    """Benchmark Kalman smoother for different linearization techniques."""

    param_names = ["linearization"]
    params = [["ekf", "ukf"]]

    def setup(self, linearization):
        dynmod, measmod, initrv, info = load_pendulum()
        _lin_method = {
            "ekf": filtsmooth.DiscreteEKFComponent,
            "ukf": filtsmooth.DiscreteUKFComponent,
        }[linearization]

        self.locations = np.arange(0.0, info["tmax"], step=info["dt"])
        _, self.observations = statespace.generate_samples(
            dynmod=dynmod, measmod=measmod, initrv=initrv, times=self.locations
        )
        self.regression_problem = problems.RegressionProblem(
            observations=self.observations, locations=self.locations
        )

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = filtsmooth.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )
        self.filtering_posterior = self.kalman_filter.filter(self.regression_problem)

    def time_smooth(self, linearization):
        self.kalman_filter.smooth(filter_posterior=self.filtering_posterior)

    def peakmem_smooth(self, linearization):
        self.kalman_filter.smooth(filter_posterior=self.filtering_posterior)


class DenseGridOperations:
    """Benchmark operations on a dense grid given the posteriors.

    That includes
        * Extrapolating / interpolating using the filter posterior
        * Extrapolating / interpolating using the smoothing posterior
        * Drawing samples from the smoothing posterior
    """

    param_names = ["linearization", "num_samples"]
    params = [["ekf", "ukf"], [1, 10]]

    def setup(self, linearization, num_samples):
        dynmod, measmod, initrv, info = load_pendulum()
        _lin_method = {
            "ekf": filtsmooth.DiscreteEKFComponent,
            "ukf": filtsmooth.DiscreteUKFComponent,
        }[linearization]

        self.locations = np.arange(0.0, info["tmax"], step=info["dt"])
        self.dense_locations = np.sort(
            np.random.uniform(
                low=0.0, high=1.2 * info["tmax"], size=int(1.2 * len(self.locations))
            )
        )

        _, self.observations = statespace.generate_samples(
            dynmod=dynmod, measmod=measmod, initrv=initrv, times=self.locations
        )
        self.regression_problem = problems.RegressionProblem(
            observations=self.observations, locations=self.locations
        )

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = filtsmooth.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )
        self.filtering_posterior = self.kalman_filter.filter(self.regression_problem)
        self.smoothing_posterior = self.kalman_filter.smooth(
            filter_posterior=self.filtering_posterior
        )

    def time_sample(self, linearization, num_samples):
        self.smoothing_posterior.sample(t=self.dense_locations, size=num_samples)

    def peakmem_sample(self, linearization, num_samples):
        self.smoothing_posterior.sample(t=self.dense_locations, size=num_samples)

    def time_dense_filter(self, linearization, num_samples):
        self.filtering_posterior(self.dense_locations)

    def peakmem_dense_filter(self, linearization, num_samples):
        self.filtering_posterior(self.dense_locations)

    def time_dense_smoother(self, linearization, num_samples):
        self.smoothing_posterior(self.dense_locations)

    def peakmem_dense_smoother(self, linearization, num_samples):
        self.smoothing_posterior(self.dense_locations)
