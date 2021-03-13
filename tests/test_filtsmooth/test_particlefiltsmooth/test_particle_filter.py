import matplotlib.pyplot as plt
import numpy as np
import pytest

from probnum import filtsmooth, random_variables, statespace


def test_effective_number_of_events():
    weights = np.random.rand(10)
    categ = random_variables.Categorical(
        support=np.random.rand(10, 2), event_probabilities=weights / np.sum(weights)
    )
    ess = filtsmooth.effective_number_of_events(categ)
    assert 0 < ess < 10


@pytest.fixture
def num_gridpoints():
    return 25


@pytest.fixture
def num_particles():
    return 15


@pytest.fixture
def data(num_gridpoints):

    locations = np.linspace(0, 2 * np.pi, num_gridpoints)
    data = 0.005 * np.random.randn(num_gridpoints) + np.sin(locations)
    return data, locations


@pytest.fixture
def setup(num_particles):
    prior = statespace.IBM(1, 1)
    measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=np.eye(1, 2),
        shift_vec=np.zeros(1),
        proc_noise_cov_mat=0.00025 * np.eye(1),
    )
    initrv = random_variables.Normal(np.zeros(2), 0.01 * np.eye(2))

    particle = filtsmooth.ParticleFilter(
        prior,
        measmod,
        initrv,
        num_particles=num_particles,
        importance_density_choice="bootstrap",
    )
    return prior, measmod, initrv, particle


def test_sth(setup, data, num_gridpoints, num_particles):
    data, locations = data
    prior, measmod, initrv, particle = setup

    posterior = particle.filter(data.reshape((-1, 1)), locations)
    states = posterior.supports
    weights = posterior.event_probabilities

    assert states.shape == (num_gridpoints, num_particles, 2)
    assert weights.shape == (num_gridpoints, num_particles)

    mean = posterior.mean
    # cov = posterior.cov
    print(np.linalg.norm(mean[:, 0] - np.sin(locations)))
    # print(cov)

    for i in range(num_particles):
        for l, p, w in zip(locations, states[:, i, 0], weights[:, i]):

            plt.plot(l, p, "o", alpha=min(0.01 + 10 * w, 1.0), color="k")

    plt.plot(locations, np.sin(locations))
    plt.plot(locations, mean[:, 0])
    plt.ylim((-2, 2))
    plt.show()

    assert False
