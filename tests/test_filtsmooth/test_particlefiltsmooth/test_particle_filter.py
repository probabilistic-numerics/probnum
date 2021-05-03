import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, randvars


def test_effective_number_of_events():
    weights = np.random.rand(10)
    categ = randvars.Categorical(
        support=np.random.rand(10, 2), probabilities=weights / np.sum(weights)
    )
    ess = filtsmooth.effective_number_of_events(categ)
    assert 0 < ess < 10


#####################################
# Test the RMSE on a pendulum example
#####################################

# Measmod style checks bootstrap and Gaussian proposals.
all_importance_distributions = pytest.mark.parametrize("measmod_style", ["ukf", "none"])

# Resampling percentage threshold checks that
# resampling is performed a) never, b) sometimes, c) always
all_resampling_configurations = pytest.mark.parametrize(
    "resampling_percentage_threshold", [-1.0, 0.1, 2.0]
)


@pytest.fixture
def num_particles():
    return 10


@pytest.fixture
def problem():
    return filtsmooth_zoo.pendulum(step=0.12)


@pytest.fixture
def particle_filter(
    problem, num_particles, measmod_style, resampling_percentage_threshold
):
    _, statespace_components = problem
    linearized_measmod = (
        filtsmooth.DiscreteUKFComponent(statespace_components["measurement_model"])
        if measmod_style == "ukf"
        else None
    )

    particle = filtsmooth.ParticleFilter(
        statespace_components["dynamics_model"],
        statespace_components["measurement_model"],
        statespace_components["initrv"],
        num_particles=num_particles,
        linearized_measurement_model=linearized_measmod,
        resampling_percentage_threshold=resampling_percentage_threshold,
    )
    return particle


@pytest.fixture()
def regression_problem(problem):
    """Filter and regression problem."""
    regression_problem, _ = problem

    return regression_problem


@all_importance_distributions
@all_resampling_configurations
def test_random_state(particle_filter):
    initrv = particle_filter.initrv
    assert initrv.random_state == particle_filter.random_state


@pytest.fixture
def pf_output(particle_filter, regression_problem):
    posterior, _ = particle_filter.filter(regression_problem)
    return posterior


@all_importance_distributions
@all_resampling_configurations
def test_shape_pf_output(pf_output, regression_problem, num_particles):
    np.random.seed(12345)

    states = pf_output.states.support
    weights = pf_output.states.probabilities
    num_gridpoints = len(regression_problem.locations)
    assert states.shape == (num_gridpoints, num_particles, 2)
    assert weights.shape == (num_gridpoints, num_particles)


@all_importance_distributions
@all_resampling_configurations
def test_rmse_particlefilter(pf_output, regression_problem):
    """Assert that the RMSE of the mode of the posterior of the PF is a lot smaller than
    the RMSE of the data."""

    np.random.seed(12345)

    true_states = regression_problem.solution

    mode = pf_output.states.mode
    rmse_mode = np.linalg.norm(np.sin(mode) - np.sin(true_states)) / np.sqrt(
        true_states.size
    )
    rmse_data = np.linalg.norm(
        regression_problem.observations - np.sin(true_states)
    ) / np.sqrt(true_states.size)

    # RMSE of PF.mode strictly better than RMSE of data
    assert rmse_mode < 0.9 * rmse_data
