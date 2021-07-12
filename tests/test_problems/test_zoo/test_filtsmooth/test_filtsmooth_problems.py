import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import problems


def rng():
    return np.random.default_rng(seed=123)


# The logistic ODE problem is not random, so it does not get a random number generator.
all_filtmooth_setups = pytest.mark.parametrize(
    "filtsmooth_setup",
    [
        filtsmooth_zoo.benes_daum(rng=rng()),
        filtsmooth_zoo.car_tracking(rng=rng()),
        filtsmooth_zoo.logistic_ode(),
        filtsmooth_zoo.ornstein_uhlenbeck(rng=rng()),
        filtsmooth_zoo.pendulum(rng=rng()),
    ],
)


@all_filtmooth_setups
def test_types(filtsmooth_setup):
    regression_problem, info = filtsmooth_setup
    assert isinstance(regression_problem, problems.TimeSeriesRegressionProblem)
    assert isinstance(regression_problem.observations, np.ndarray)
    assert isinstance(regression_problem.locations, np.ndarray)
    assert isinstance(regression_problem.solution, np.ndarray)
    assert isinstance(info, dict)
    assert "prior_process" in info


@all_filtmooth_setups
def test_shapes(filtsmooth_setup):
    regression_problem, info = filtsmooth_setup
    assert regression_problem.locations.size == regression_problem.observations.shape[0]
    mm = regression_problem.measurement_models[0]
    real = info["prior_process"].initrv.mean
    measurement = mm.forward_realization(real, t=1.0)[0]
    assert measurement.shape == regression_problem.observations[0].shape
