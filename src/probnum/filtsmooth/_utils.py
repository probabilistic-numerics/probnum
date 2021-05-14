"""Utility functions for filtering and smoothing."""


import numpy as np

from probnum import problems

__all__ = ["merge_regression_problems"]


def merge_regression_problems(problem_and_likelihood1, problem_and_likelihood2):
    """Make a new regression problem out of two other regression problems.

    Parameters
    ----------
    problem_and_likelihood1 :
        Tuple of a RegressionProblem and an array of Transitions.
    problem_and_likelihood2 :
        Tuple of a RegressionProblem and an array of Transitions.

    Returns
    -------
    Tuple of a RegressionProblem and an array of Transitions that merges locations, data, and measmods of both problems. The output is sorted according to the locations.

    Note
    ----
    To merge more than two problems, combine this function with functools.reduce.
    """

    regression_problem1, measurement_models1 = problem_and_likelihood1
    regression_problem2, measurement_models2 = problem_and_likelihood2

    # Check input types of measurement models: we explicitly want numpy arrays.
    # More abstract merging (e.g. of lists or generators) is not supported currently.
    errormsg = (
        f"The measurement models are expected to be of type `{np.ndarray}'"
        f"but type `{type(measurement_models1)}' was received."
    )
    if not isinstance(measurement_models1, np.ndarray):
        raise TypeError(errormsg)
    if not isinstance(measurement_models2, np.ndarray):
        raise TypeError(errormsg)

    # Some shorthand improves readibility of the inserts below.
    locs1, data1, sol1 = (
        regression_problem1.locations,
        regression_problem1.observations,
        regression_problem1.solution,
    )
    locs2, data2, sol2 = (
        regression_problem2.locations,
        regression_problem2.observations,
        regression_problem2.solution,
    )

    # Merge time locations
    if np.any(np.in1d(locs1, locs2)):
        raise ValueError("Regression problems must not share time locations.")
    new_locs = np.sort(np.concatenate((locs1, locs2)))
    locs1_in_new_locs = np.searchsorted(new_locs, locs1)
    locs2_in_new_locs = np.searchsorted(new_locs, locs2)

    # Merge observations
    new_num_obs = len(data1) + len(data2)
    if not data1.shape[1:] == data2.shape[1:]:
        raise ValueError("The data sets have incompatible dimension.")
    new_data_shape = (new_num_obs,) + data1.shape[1:]
    new_data = np.zeros(new_data_shape)
    new_data[locs1_in_new_locs] = data1
    new_data[locs2_in_new_locs] = data2

    # Merge solutions
    if sol1 is not None and sol2 is not None:
        if not sol1.shape[1:] == sol2.shape[1:]:
            raise ValueError("The solution arrays have incompatible dimension.")
        new_sol_shape = (new_num_obs,) + sol1.shape[1:]
        new_sol = np.zeros(new_sol_shape)
        new_sol[locs1_in_new_locs] = sol1
        new_sol[locs2_in_new_locs] = sol2
    else:
        new_sol = None

    # Merge measurement models
    new_measurement_models = np.zeros((new_num_obs,), dtype=object)
    new_measurement_models[locs1_in_new_locs] = measurement_models1
    new_measurement_models[locs2_in_new_locs] = measurement_models2

    # Return merged arrays
    new_regression_problem = problems.RegressionProblem(
        locations=new_locs, observations=new_data, solution=new_sol
    )
    return new_regression_problem, new_measurement_models
