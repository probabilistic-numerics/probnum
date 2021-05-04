import functools
import itertools

import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, problems


@pytest.fixture
def car_tracking1():
    return filtsmooth_zoo.car_tracking(
        measurement_variance=2.0, timespan=(0.0, 10.0), step=0.5
    )


@pytest.fixture
def car_tracking2():
    return filtsmooth_zoo.car_tracking(
        measurement_variance=0.5, timespan=(0.1, 10.1), step=0.5
    )


@pytest.fixture
def car_tracking3():
    return filtsmooth_zoo.car_tracking(
        measurement_variance=1.23, timespan=(0.2, 10.2), step=0.5
    )


def test_merge_regression_problems(car_tracking1, car_tracking2):
    """Regression problems yield the correct shapes."""
    prob1, info1 = car_tracking1
    measmod1 = np.asarray([info1["measurement_model"]] * len(prob1.locations))

    prob2, info2 = car_tracking2
    measmod2 = np.asarray([info2["measurement_model"]] * len(prob2.locations))
    new_prob, new_models = filtsmooth.merge_regression_problems(
        (prob1, measmod1), (prob2, measmod2)
    )

    N = len(prob1.locations) + len(prob2.locations)
    d1 = prob1.solution.shape[1:]
    d2 = prob1.observations.shape[1:]

    assert isinstance(new_prob, problems.RegressionProblem)
    assert new_prob.locations.shape == (N,)
    assert new_prob.solution.shape == (N,) + d1
    assert new_prob.observations.shape == (N,) + d2

    assert isinstance(new_models, np.ndarray)
    assert new_models.shape == (N,)


def test_merge_raises_type_error(car_tracking1, car_tracking2):
    prob1, info1 = car_tracking1
    prob2, info2 = car_tracking2

    # Single models are rejected
    with pytest.raises(TypeError):
        measmod1 = info1["measurement_model"]
        measmod2 = info2["measurement_model"]
        filtsmooth.merge_regression_problems((prob1, measmod1), (prob2, measmod2))

    # Lists are rejected
    with pytest.raises(TypeError):
        measmod1 = [info1["measurement_model"]] * len(prob1.locations)
        measmod2 = [info2["measurement_model"]] * len(prob2.locations)
        filtsmooth.merge_regression_problems((prob1, measmod1), (prob2, measmod2))

    # Tuples are rejected
    with pytest.raises(TypeError):
        measmod1 = (info1["measurement_model"]) * len(prob1.locations)
        measmod2 = (info2["measurement_model"]) * len(prob2.locations)
        filtsmooth.merge_regression_problems((prob1, measmod1), (prob2, measmod2))

    # Generators are rejected
    with pytest.raises(TypeError):
        measmod1 = itertools.repeat(info1["measurement_model"], n=len(prob1.locations))
        measmod2 = itertools.repeat(info2["measurement_model"], n=len(prob2.locations))
        filtsmooth.merge_regression_problems((prob1, measmod1), (prob2, measmod2))


def test_merge_works_with_reduce(car_tracking1, car_tracking2, car_tracking3):

    prob1, info1 = car_tracking1
    measmod1 = np.asarray([info1["measurement_model"]] * len(prob1.locations))

    prob2, info2 = car_tracking2
    measmod2 = np.asarray([info2["measurement_model"]] * len(prob2.locations))

    prob3, info3 = car_tracking3
    measmod3 = np.asarray([info3["measurement_model"]] * len(prob3.locations))

    new_prob, new_models = functools.reduce(
        filtsmooth.merge_regression_problems,
        ((prob1, measmod1), (prob2, measmod2), (prob3, measmod3)),
    )

    N = len(prob1.locations) + len(prob2.locations) + len(prob3.locations)
    d1 = prob1.solution.shape[1:]
    d2 = prob1.observations.shape[1:]

    assert isinstance(new_prob, problems.RegressionProblem)
    assert new_prob.locations.shape == (N,)
    assert new_prob.solution.shape == (N,) + d1
    assert new_prob.observations.shape == (N,) + d2

    assert isinstance(new_models, np.ndarray)
    assert new_models.shape == (N,)


def test_shared_locations_raise_error(car_tracking1):

    prob1, info1 = car_tracking1
    measmod1 = np.asarray([info1["measurement_model"]] * len(prob1.locations))

    with pytest.raises(ValueError):
        filtsmooth.merge_regression_problems((prob1, measmod1), (prob1, measmod1))
