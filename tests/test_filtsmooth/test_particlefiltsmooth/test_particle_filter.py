import matplotlib.pyplot as plt
import numpy as np
import pytest

from probnum import filtsmooth, random_variables, statespace


@pytest.fixture
def num_gridpoints():
    return 50


@pytest.fixture
def num_particles():
    return 100


@pytest.fixture
def data(num_gridpoints):

    locations = np.linspace(0, 2 * np.pi, num_gridpoints)
    data = 0.05 * np.random.randn(num_gridpoints) + np.sin(locations)
    return data, locations


@pytest.fixture
def setup(num_particles):
    prior = statespace.IBM(1, 1)
    measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=np.eye(1, 2),
        shift_vec=np.zeros(1),
        proc_noise_cov_mat=0.025 * np.eye(1),
    )
    initrv = random_variables.Normal(np.zeros(2), 0.01 * np.eye(2))

    particle = filtsmooth.ParticleFilter(
        prior, measmod, initrv, num_particles=num_particles
    )
    return prior, measmod, initrv, particle


def test_sth(setup, data, num_gridpoints, num_particles):
    data, locations = data
    prior, measmod, initrv, particle = setup

    posterior = particle.filter(data.reshape((-1, 1)), locations)
    states = np.array([state.particles for state in posterior.particle_state_list])
    weights = np.array([state.weights for state in posterior.particle_state_list])
    assert states.shape == (num_gridpoints, num_particles, 2)
    assert weights.shape == (num_gridpoints, num_particles)

    for i in range(10):
        for l, p, w in zip(locations, states[:, i, 0], weights[:, i]):

            plt.plot(l, p, "o", alpha=min(0.01 + 2 * w, 1.0), color="k")
    plt.plot(locations, np.sin(locations))
    plt.ylim((-2, 2))
    plt.show()
    assert False
