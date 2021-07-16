"""Utility functions for differential equation problems."""

from typing import Sequence, Union

import numpy as np

from probnum import problems, statespace
from probnum.diffeq import information_operators


def ivp_to_regression_problem(
    ivp: problems.InitialValueProblem,
    locations: Union[Sequence, np.ndarray],
    ode_information_operator: information_operators.ODEInformationOperator,
):

    N = len(locations)
    data = np.zeros((N + 1, ivp.dimension))
    if ivp.solution is not None:
        solution = [ivp.solution(t) for t in locations]
    else:
        solution = None

    # Assemble measurement models
    transition_matrix = np.eye(ode_information_operator.input_dim, ivp.dimension)
    shift_vector = -ivp.y0
    measmod_initial_condition = statespace.DiscreteLTIGaussian.from_linop(
        state_trans_mat=transition_matrix, shift_vec=shift_vector
    )
    measmod_ode = [ode_information_operator.as_transition()] * N
    measmod_list = measmod_initial_condition + measmod_ode

    return problems.TimeSeriesRegressionProblem(
        locations=grid,
        observations=data,
        measurement_models=measmod_list,
        solution=solution,
    )
