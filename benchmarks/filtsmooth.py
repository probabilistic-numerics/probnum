"""Benchmarks for Gaussian filtering."""
import numpy as np

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum.randvars import Normal


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
    dynamod = pnss.DiscreteGaussian(2, 2, f, lambda t: q, df)
    measmod = pnss.DiscreteGaussian(2, 1, h, lambda t: r, dh)
    initrv = Normal(initmean, initcov)
    return dynamod, measmod, initrv, {"dt": delta_t, "tmax": 4}


class Filtering:
    """Benchmark Kalman filter for different linearization techniques."""

    param_names = ["linearization"]
    params = [["ekf", "ukf"]]

    def setup(self, linearization):
        dynmod, measmod, initrv, info = load_pendulum()
        _lin_method = {
            "ekf": pnfs.DiscreteEKFComponent,
            "ukf": pnfs.DiscreteUKFComponent,
        }[linearization]

        self.locations = np.arange(0.0, info["tmax"], step=info["dt"])
        _, self.observations = pnss.generate_samples(
            dynmod=dynmod, measmod=measmod, initrv=initrv, times=self.locations
        )

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = pnfs.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )

    def time_filter(self, linearization):
        self.kalman_filter.filter(dataset=self.observations, times=self.locations)

    def peakmem_filter(self, linearization):
        self.kalman_filter.filter(dataset=self.observations, times=self.locations)


class Smoothing:
    """Benchmark Kalman filter for different linearization techniques."""

    param_names = ["linearization"]
    params = [["ekf", "ukf"]]

    def setup(self, linearization):
        dynmod, measmod, initrv, info = load_pendulum()
        _lin_method = {
            "ekf": pnfs.DiscreteEKFComponent,
            "ukf": pnfs.DiscreteUKFComponent,
        }[linearization]

        self.locations = np.arange(0.0, info["tmax"], step=info["dt"])
        _, self.observations = pnss.generate_samples(
            dynmod=dynmod, measmod=measmod, initrv=initrv, times=self.locations
        )

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = pnfs.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )
        self.filtering_posterior = self.kalman_filter.filter(
            dataset=self.observations, times=self.locations
        )

    def time_smooth(self, linearization):
        self.kalman_filter.smooth(filter_posterior=self.filtering_posterior)

    def peakmem_smooth(self, linearization):
        self.kalman_filter.smooth(filter_posterior=self.filtering_posterior)


class Sampling:
    """Benchmark Kalman filter for different linearization techniques."""

    param_names = ["linearization", "num_samples"]
    params = [["ekf", "ukf"], [1, 10]]

    def setup(self, linearization, num_samples):
        dynmod, measmod, initrv, info = load_pendulum()
        _lin_method = {
            "ekf": pnfs.DiscreteEKFComponent,
            "ukf": pnfs.DiscreteUKFComponent,
        }[linearization]

        self.locations = np.arange(0.0, info["tmax"], step=info["dt"])
        _, self.observations = pnss.generate_samples(
            dynmod=dynmod, measmod=measmod, initrv=initrv, times=self.locations
        )

        linearized_dynmod = _lin_method(dynmod)
        linearized_measmod = _lin_method(measmod)

        self.kalman_filter = pnfs.Kalman(
            dynamics_model=linearized_dynmod,
            measurement_model=linearized_measmod,
            initrv=initrv,
        )
        filtering_posterior = self.kalman_filter.filter(
            dataset=self.observations, times=self.locations
        )
        self.smoothing_posterior = self.kalman_filter.smooth(
            filter_posterior=filtering_posterior
        )

    def time_sample(self, linearization, num_samples):
        self.smoothing_posterior.sample(size=num_samples)

    def peakmem_sample(self, linearization, num_samples):
        self.smoothing_posterior.sample(size=num_samples)
