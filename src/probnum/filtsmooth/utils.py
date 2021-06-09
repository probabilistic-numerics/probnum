"""Utility functions for filtering and smoothing."""


from typing import Tuple

import numpy as np

from probnum import problems

__all__ = ["merge_regression_problems"]


def merge_regression_problems(
    problem_and_likelihood1: Tuple[problems.RegressionProblem, np.ndarray],
    problem_and_likelihood2: Tuple[problems.RegressionProblem, np.ndarray],
) -> Tuple[problems.RegressionProblem, np.ndarray]:
    """Make a new regression problem out of two other regression problems.

    Parameters
    ----------
    problem_and_likelihood1 :
        Tuple of a RegressionProblem and an array of Transitions.
    problem_and_likelihood2 :
        Tuple of a RegressionProblem and an array of Transitions.

    Raises
    ------
    TypeError
        If the lists of measurement models are not numpy arrays.
        This is required (for the moment) in order to simplify the implementation.
        More general merging (e.g. of lists or generators) is not supported currently.
    ValueError
        If the locations in both regression problems are not disjoint.
        Multiple observations at a single grid point are not supported currently.

    Returns
    -------
    Tuple of a RegressionProblem and an array of Transitions that merges locations, data, and measmods of both problems. The output is sorted according to the locations.

    Note
    ----
    To merge more than two problems, combine this function with functools.reduce.

    Examples
    --------

    Create two car-tracking problems with similar parameters and disjoint locations.

    >>> import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
    >>> prob1, info1 = filtsmooth_zoo.car_tracking(
    ...     measurement_variance=2.0, timespan=(0.0, 10.0), step=0.5
    ... )
    >>> measmods1 = np.asarray([info1["measurement_model"]] * len(prob1.locations))
    >>> print(prob1.locations)
    [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5
     9.  9.5]

    >>> prob2, info2 = filtsmooth_zoo.car_tracking(
    ...     measurement_variance=2.0, timespan=(0.25, 10.25), step=0.5
    ... )
    >>> measmods2 = np.asarray([info2["measurement_model"]] * len(prob2.locations))
    >>> print(prob2.locations)
    [0.25 0.75 1.25 1.75 2.25 2.75 3.25 3.75 4.25 4.75 5.25 5.75 6.25 6.75
     7.25 7.75 8.25 8.75 9.25 9.75]

    Merge them with merge_regression_problems

    >>> new_prob, new_models = merge_regression_problems(
    ...     (prob1, measmods1), (prob2, measmods2)
    ... )
    >>> print(new_prob.locations)
    [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.   2.25 2.5  2.75 3.   3.25
     3.5  3.75 4.   4.25 4.5  4.75 5.   5.25 5.5  5.75 6.   6.25 6.5  6.75
     7.   7.25 7.5  7.75 8.   8.25 8.5  8.75 9.   9.25 9.5  9.75]

    If you have more than two problems that you want to merge, do this with functools.reduce.

    >>> import functools
    >>> prob3, info3 = filtsmooth_zoo.car_tracking(
    ...     measurement_variance=2.0, timespan=(0.35, 10.35), step=0.5
    ... )
    >>> measmods3 = np.asarray([info3["measurement_model"]] * len(prob3.locations))
    >>> new_prob, new_models = functools.reduce(
    ...     merge_regression_problems,
    ...     ((prob1, measmods1), (prob2, measmods2), (prob3, measmods3)),
    ... )
    >>> print(new_prob.locations)
    [0.   0.25 0.35 0.5  0.75 0.85 1.   1.25 1.35 1.5  1.75 1.85 2.   2.25
     2.35 2.5  2.75 2.85 3.   3.25 3.35 3.5  3.75 3.85 4.   4.25 4.35 4.5
     4.75 4.85 5.   5.25 5.35 5.5  5.75 5.85 6.   6.25 6.35 6.5  6.75 6.85
     7.   7.25 7.35 7.5  7.75 7.85 8.   8.25 8.35 8.5  8.75 8.85 9.   9.25
     9.35 9.5  9.75 9.85]
    """

    regression_problem1, measurement_models1 = problem_and_likelihood1
    regression_problem2, measurement_models2 = problem_and_likelihood2

    # Check input types of measurement models: we explicitly want numpy arrays.
    # More abstract merging (e.g. of lists or generators) is not supported currently.
    errormsg_type_measmods = (
        f"The measurement models are expected to be of type `{np.ndarray}'"
        f"but type `{type(measurement_models1)}' was received."
    )
    if not isinstance(measurement_models1, np.ndarray):
        raise TypeError(errormsg_type_measmods)
    if not isinstance(measurement_models2, np.ndarray):
        raise TypeError(errormsg_type_measmods)

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

    # Merge solutions.
    # The resulting problem will only have a solution of BOTH problems have one.
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
