"""Utility functions for differential equation problems."""

from typing import Optional, Sequence, Union

import numpy as np

from probnum import problems, statespace
from probnum.diffeq.odefiltsmooth import approx_strategies, information_operators
from probnum.typing import FloatArgType

__all__ = ["ivp_to_regression_problem"]


def ivp_to_regression_problem(
    ivp: problems.InitialValueProblem,
    locations: Union[Sequence, np.ndarray],
    ode_information_operator: information_operators.InformationOperator,
    approx_strategy: Optional[approx_strategies.ApproximationStrategy] = None,
    ode_measurement_variance: Optional[FloatArgType] = None,
):
    """Transform an initial value problem into a regression problem."""

    # Construct data and solution
    N = len(locations)
    data = np.zeros((N, ivp.dimension))
    if ivp.solution is not None:
        solution = [ivp.solution(t) for t in locations]
    else:
        solution = None

    # Construct measurement models
    measmod_initial_condition, measmod_ode = _construct_measurement_models(
        ivp, ode_information_operator, approx_strategy, ode_measurement_variance
    )
    measmod_list = [measmod_initial_condition] + [measmod_ode] * (N - 1)

    # Return regression problem
    return problems.TimeSeriesRegressionProblem(
        locations=locations,
        observations=data,
        measurement_models=measmod_list,
        solution=solution,
    )


def _construct_measurement_models(
    ivp, ode_information_operator, approx_strategy, ode_measurement_variance
):
    """Construct measurement models for the IVP."""

    transition_matrix = np.eye(
        ode_information_operator.output_dim, ode_information_operator.input_dim
    )
    shift_vector = -ivp.y0

    if ode_measurement_variance is None:
        measmod_y0, measmod_ode = _construct_measurement_models_dirac_likelihood(
            ode_information_operator,
            shift_vector,
            transition_matrix,
            approx_strategy,
        )
    else:
        measmod_y0, measmod_ode = _construct_measurement_models_gaussian_likelihood(
            ode_information_operator,
            shift_vector,
            transition_matrix,
            approx_strategy,
            ode_measurement_variance,
        )
    return measmod_y0, measmod_ode


def _construct_measurement_models_gaussian_likelihood(
    ode_information_operator,
    shift_vector,
    transition_matrix,
    approx_strategy,
    ode_measurement_variance,
):
    """Construct measurement models for the IVP with Gaussian likelihoods."""

    def diff(t):
        return ode_measurement_variance * np.eye(ode_information_operator.output_dim)

    def diff_cholesky(t):
        return np.sqrt(ode_measurement_variance) * np.eye(
            ode_information_operator.output_dim
        )

    measmod_initial_condition = statespace.DiscreteLTIGaussian(
        state_trans_mat=transition_matrix,
        shift_vec=shift_vector,
        proc_noise_cov_mat=diff(None),
        proc_noise_cov_cholesky=diff_cholesky(None),
    )
    if approx_strategy is not None:
        ode_information_operator = approx_strategy(ode_information_operator)
    measmod_ode = ode_information_operator.as_transition(
        measurement_cov_fun=diff, measurement_cov_cholesky_fun=diff_cholesky
    )

    return measmod_initial_condition, measmod_ode


def _construct_measurement_models_dirac_likelihood(
    ode_information_operator, shift_vector, transition_matrix, approx_strategy
):
    """Construct measurement models for the IVP with Dirac likelihoods."""
    measmod_initial_condition = statespace.DiscreteLTIGaussian.from_linop(
        state_trans_mat=transition_matrix, shift_vec=shift_vector
    )
    if approx_strategy is not None:
        ode_information_operator = approx_strategy(ode_information_operator)
    measmod_ode = ode_information_operator.as_transition()

    return measmod_initial_condition, measmod_ode
