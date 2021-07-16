import functools

import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, problems


@pytest.fixture
def car_tracking1(rng):
    return filtsmooth_zoo.car_tracking(
        rng=rng, measurement_variance=2.0, timespan=(0.0, 10.0), step=0.5
    )


@pytest.fixture
def car_tracking2(rng):
    return filtsmooth_zoo.car_tracking(
        rng=rng, measurement_variance=0.5, timespan=(0.1, 10.1), step=0.5
    )


@pytest.fixture
def car_tracking3(rng):
    return filtsmooth_zoo.car_tracking(
        rng=rng, measurement_variance=1.23, timespan=(0.2, 10.2), step=0.5
    )


def test_merge_regression_problems(car_tracking1, car_tracking2):
    """Regression problems yield the correct shapes."""
    prob1, info1 = car_tracking1
    prob2, info2 = car_tracking2
    new_prob = filtsmooth.utils.merge_regression_problems(prob1, prob2)

    N = len(prob1.locations) + len(prob2.locations)
    d1 = prob1.solution.shape[1:]
    d2 = prob1.observations.shape[1:]

    assert isinstance(new_prob, problems.TimeSeriesRegressionProblem)
    assert new_prob.locations.shape == (N,)
    assert new_prob.solution.shape == (N,) + d1
    assert new_prob.observations.shape == (N,) + d2

    assert isinstance(new_prob.measurement_models, np.ndarray)
    assert new_prob.measurement_models.shape == (N,)


def test_merge_works_with_reduce(car_tracking1, car_tracking2, car_tracking3):
    """Assert that the merge function is compatible with functools.reduce."""
    prob1, info1 = car_tracking1

    prob2, info2 = car_tracking2

    prob3, info3 = car_tracking3

    new_prob = functools.reduce(
        filtsmooth.utils.merge_regression_problems,
        (prob1, prob2, prob3),
    )

    N = len(prob1.locations) + len(prob2.locations) + len(prob3.locations)
    d1 = prob1.solution.shape[1:]
    d2 = prob1.observations.shape[1:]

    assert isinstance(new_prob, problems.TimeSeriesRegressionProblem)
    assert new_prob.locations.shape == (N,)
    assert new_prob.solution.shape == (N,) + d1
    assert new_prob.observations.shape == (N,) + d2

    assert isinstance(new_prob.measurement_models, np.ndarray)
    assert new_prob.measurement_models.shape == (N,)


def test_shared_locations_raise_error(car_tracking1):
    """Assert that both problems are not allowed to share locations."""
    prob1, _ = car_tracking1

    with pytest.raises(ValueError):
        filtsmooth.utils.merge_regression_problems(prob1, prob1)


def test_data_sets_incompatible_dimensions(car_tracking1, car_tracking2):
    prob1, _ = car_tracking1
    prob2, _ = car_tracking2

    # Change the dimension of the data of one of the problems
    prob1.observations = prob1.observations[:, 1:]

    with pytest.raises(ValueError):
        filtsmooth.utils.merge_regression_problems(prob1, prob2)


def test_solutions_incompatible_dimensions(car_tracking1, car_tracking2):
    prob1, _ = car_tracking1
    prob2, _ = car_tracking2

    # Change the dimension of the data of one of the problems
    prob1.solution = prob1.solution[:, 1:]

    with pytest.raises(ValueError):
        filtsmooth.utils.merge_regression_problems(prob1, prob2)


def test_solutions_not_available(car_tracking1, car_tracking2):
    """As soon as ONE of the problems does not have a solution, the merged problem does
    not have one."""
    prob1, _ = car_tracking1
    prob2, _ = car_tracking2

    # Sanity check: in principle, the output would have a solution
    prob = filtsmooth.utils.merge_regression_problems(prob1, prob2)
    assert prob.solution is not None

    # Removing one of the solutions makes the output have no solution
    prob1.solution = None
    prob = filtsmooth.utils.merge_regression_problems(prob1, prob2)
    assert prob.solution is None
