import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, randvars


def test_effective_number_of_events():
    weights = np.random.rand(10)
    categ = randvars.Categorical(
        support=np.random.rand(10, 2), probabilities=weights / np.sum(weights)
    )
    ess = filtsmooth.particle.effective_number_of_events(categ)
    assert 0 < ess < 10


#####################################
# Test the RMSE on a pendulum example
#####################################

# Measmod style checks bootstrap and Gaussian proposals.
all_importance_distributions = pytest.mark.parametrize(
    "measmod_style", ["ukf", "ekf", "none"]
)

# Resampling percentage threshold checks that
# resampling is performed a) never, b) sometimes, c) always
all_resampling_configurations = pytest.mark.parametrize(
    "resampling_percentage_threshold", [-1.0, 0.1, 2.0]
)


@pytest.fixture
def num_particles():
    return 30


# Parameterize initarg with None and -10 to test both initial setups for the PF:
# if None, the initrv is processed (through the importance distribution)
# before sampling an initial set of particles.
# If -10, the initial set of particles is sampled immediately.
@pytest.fixture(params=[None, -10])
def problem(request, rng):
    initarg = request.param
    return filtsmooth_zoo.pendulum(rng=rng, step=0.12, initarg=initarg)


@pytest.fixture
def particle_filter_setup(
    problem, num_particles, measmod_style, resampling_percentage_threshold, rng
):
    _, info = problem
    prior_process = info["prior_process"]
    if measmod_style == "ekf":
        importance_distribution = (
            filtsmooth.particle.LinearizationImportanceDistribution.from_ukf(
                prior_process.transition
            )
        )
    elif measmod_style == "ukf":
        importance_distribution = (
            filtsmooth.particle.LinearizationImportanceDistribution.from_ekf(
                prior_process.transition
            )
        )
    else:
        importance_distribution = filtsmooth.particle.BootstrapImportanceDistribution(
            prior_process.transition
        )
    particle = filtsmooth.particle.ParticleFilter(
        prior_process,
        importance_distribution=importance_distribution,
        num_particles=num_particles,
        rng=rng,
        resampling_percentage_threshold=resampling_percentage_threshold,
    )
    return particle


@pytest.fixture()
def regression_problem(problem):
    """Filter and regression problem."""
    regression_problem, *_ = problem

    return regression_problem


@pytest.fixture
def pf_output(particle_filter_setup, regression_problem):
    particle_filter = particle_filter_setup
    posterior, _ = particle_filter.filter(regression_problem)
    return posterior


@all_importance_distributions
@all_resampling_configurations
def test_shape_pf_output(pf_output, regression_problem, num_particles):

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

    true_states = regression_problem.solution

    mode = pf_output.states.mode
    rmse_mode = np.linalg.norm(np.sin(mode) - np.sin(true_states)) / np.sqrt(
        true_states.size
    )
    rmse_data = np.linalg.norm(
        regression_problem.observations - np.sin(true_states)
    ) / np.sqrt(true_states.size)

    # RMSE of PF.mode strictly better than RMSE of data
    assert rmse_mode < 0.99 * rmse_data
