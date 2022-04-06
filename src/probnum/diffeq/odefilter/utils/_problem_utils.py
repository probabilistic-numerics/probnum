"""Utility functions for differential equation problems."""

from typing import Optional, Sequence, Union

import numpy as np

from probnum import problems, randprocs, randvars
from probnum.diffeq.odefilter import approx_strategies, information_operators
from probnum.typing import FloatLike

__all__ = ["ivp_to_regression_problem"]

# The ODE information operator is not optional, because in order to create it
# one needs to know the order of the algorithm that is desired
# (i.e. num_prior_derivatives). Since this is a weird input for the function, it seems
# safer to just require the full operator.
def ivp_to_regression_problem(
    ivp: problems.InitialValueProblem,
    locations: Union[Sequence, np.ndarray],
    ode_information_operator: information_operators.InformationOperator,
    approx_strategy: Optional[approx_strategies.ApproximationStrategy] = None,
    ode_measurement_variance: Optional[FloatLike] = 0.0,
    exclude_initial_condition=False,
):
    """Transform an initial value problem into a regression problem.

    Parameters
    ----------
    ivp
        Initial value problem to be transformed.
    locations
        Locations of the time-grid-points.
    ode_information_operator
        ODE information operator to use.
    approx_strategy
        Approximation strategy to use. Optional. Default is `EK1()`.
    ode_measurement_variance
        Artificial ODE measurement noise. Optional. Default is 0.0.
    exclude_initial_condition
        Whether to exclude the initial condition from the regression problem.
        Optional. Default is False, in which case the returned measurement model list
        consist of [`initcond_mm`, `ode_mm`, ..., `ode_mm`].

    Returns
    -------
    problems.TimeSeriesRegressionProblem
        Time-series regression problem.
    """

    # Construct data and solution
    N = len(locations)
    data = np.zeros((N, ivp.dimension))
    if ivp.solution is not None:
        solution = np.stack([ivp.solution(t) for t in locations])
    else:
        solution = None

    ode_information_operator.incorporate_ode(ivp)

    # Construct measurement models
    measmod_initial_condition, measmod_ode = _construct_measurement_models(
        ivp, ode_information_operator, approx_strategy, ode_measurement_variance
    )

    if exclude_initial_condition:
        measmod_list = [measmod_ode] * N
    else:
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

    shape = ode_information_operator.output_dim, ode_information_operator.input_dim

    # Make a proj2coord(0)-like matrix without relying on the prior_process
    transition_matrix = (
        np.eye(*shape).reshape((-1,), order="C").reshape(shape, order="F")
    )
    shift_vector = -ivp.y0

    if ode_measurement_variance > 0.0:
        measmod_y0, measmod_ode = _construct_measurement_models_gaussian_likelihood(
            ode_information_operator,
            shift_vector,
            transition_matrix,
            approx_strategy,
            ode_measurement_variance,
        )
    else:
        measmod_y0, measmod_ode = _construct_measurement_models_dirac_likelihood(
            ode_information_operator,
            shift_vector,
            transition_matrix,
            approx_strategy,
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
    diff = ode_measurement_variance * np.eye(ode_information_operator.output_dim)
    diff_cholesky = np.sqrt(diff)
    noise = randvars.Normal(mean=shift_vector, cov=diff, cov_cholesky=diff_cholesky)

    measmod_initial_condition = randprocs.markov.discrete.LTIGaussian(
        transition_matrix=transition_matrix,
        noise=noise,
    )
    if approx_strategy is not None:
        ode_information_operator = approx_strategy(ode_information_operator)
    measmod_ode = ode_information_operator.as_transition(
        noise_fun=lambda t: noise,
    )

    return measmod_initial_condition, measmod_ode


def _construct_measurement_models_dirac_likelihood(
    ode_information_operator, shift_vector, transition_matrix, approx_strategy
):
    """Construct measurement models for the IVP with Dirac likelihoods."""
    measmod_initial_condition = randprocs.markov.discrete.LTIGaussian.from_linop(
        transition_matrix=transition_matrix, noise_mean=shift_vector
    )
    if approx_strategy is not None:
        ode_information_operator = approx_strategy(ode_information_operator)
    measmod_ode = ode_information_operator.as_transition()

    return measmod_initial_condition, measmod_ode
