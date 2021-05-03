"""Utility functions for filtering and smoothing."""

import collections
import itertools

import numpy as np

from probnum import problems


def merge_regression_problems(
    regression_problem1, measurement_models1, regression_problem2, measurement_models2
):
    """Make a new regression problem out of two other regression problems."""

    errormsg = (
        f"The measurement models are expected to be of type `{np.ndarray}'"
        f"but type `{type(measurement_models1)}' was received."
    )
    if not isinstance(measurement_models1, np.ndarray):
        raise TypeError(errormsg)
    if not isinstance(measurement_models2, np.ndarray):
        raise TypeError(errormsg)

    measurement_models1 = np.asarray(measurement_models1)
    measurement_models2 = np.asarray(measurement_models2)

    t1, y1 = regression_problem1.locations, regression_problem1.observations
    t2, y2 = regression_problem2.locations, regression_problem2.observations

    t = np.union1d(t1, t2)
    t1_in_t = np.searchsorted(t, t1)
    t2_in_t = np.searchsorted(t, t2)

    assert y1.shape[1:] == y2.shape[1:]
    new_shape = (len(y1) + len(y2),) + y1.shape[1:]
    y = np.zeros(new_shape)
    y[t1_in_t] = y1
    y[t2_in_t] = y2

    if (
        regression_problem1.solution is not None
        and regression_problem2.solution is not None
    ):
        assert (
            regression_problem1.solution.shape[1:]
            == regression_problem1.solution.shape[1:]
        )
        new_shape = (
            len(regression_problem1.solution) + len(regression_problem2.solution),
        ) + regression_problem2.solution.shape[1:]
        new_solution = np.zeros(new_shape)
        new_solution[t1_in_t] = regression_problem1.solution
        new_solution[t2_in_t] = regression_problem2.solution
    else:
        new_solution = None

    # Chain the measurement models
    measurement_models = np.zeros((len(y1) + len(y2)), dtype=object)
    measurement_models[t1_in_t] = measurement_models1
    measurement_models[t2_in_t] = measurement_models2
    return (
        problems.RegressionProblem(locations=t, observations=y, solution=new_solution),
        measurement_models,
    )
