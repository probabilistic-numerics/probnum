import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import problems

all_filtmooth_setups = pytest.mark.parametrize(
    "filtsmooth_setup",
    [
        filtsmooth_zoo.benes_daum(),
        filtsmooth_zoo.car_tracking(),
        filtsmooth_zoo.logistic_ode(),
        filtsmooth_zoo.ornstein_uhlenbeck(),
        filtsmooth_zoo.pendulum(),
    ],
)


@all_filtmooth_setups
def test_types(filtsmooth_setup):
    regression_problem, statespace_components = filtsmooth_setup
    assert isinstance(regression_problem, problems.RegressionProblem)
    assert isinstance(regression_problem.observations, np.ndarray)
    assert isinstance(regression_problem.locations, np.ndarray)
    assert isinstance(regression_problem.solution, np.ndarray)
    assert isinstance(statespace_components, dict)
    assert "dynamics_model" in statespace_components
    assert "measurement_model" in statespace_components
    assert "initrv" in statespace_components


@all_filtmooth_setups
def test_shapes(filtsmooth_setup):
    regression_problem, statespace_components = filtsmooth_setup
    assert regression_problem.locations.size == regression_problem.observations.shape[0]

    measurement = statespace_components["measurement_model"].forward_realization(
        statespace_components["initrv"].mean, t=1.0
    )[0]
    assert measurement.shape == regression_problem.observations[0].shape
