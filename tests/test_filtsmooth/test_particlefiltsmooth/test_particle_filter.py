import matplotlib.pyplot as plt
import numpy as np
import pytest

from probnum import filtsmooth, random_variables, statespace


@pytest.fixture
def data():
    locations = np.linspace(0, 2 * np.pi, 20)
    data = 0.5 * np.random.randn(20) + np.sin(locations)
    return data, locations


@pytest.fixture
def setup():
    prior = statespace.IBM(1, 1)
    measmod = statespace.DiscreteLTIGaussian(
        state_trans_mat=np.eye(1, 2),
        shift_vec=np.zeros(1),
        proc_noise_cov_mat=0.25 * np.eye(1),
    )
    initrv = random_variables.Normal(np.zeros(2), np.eye(2))

    particle = filtsmooth.ParticleFilter(prior, measmod, initrv, num_particles=10)
    return prior, measmod, initrv, particle


def test_sth(setup, data):
    data, locations = data
    prior, measmod, initrv, particle = setup

    posterior = particle.filter(data.reshape((-1, 1)), locations)
    states = np.array([state.particles for state in posterior.particle_state_list])
    weights = np.array([state.weights for state in posterior.particle_state_list])
    assert states.shape == (20, 10, 2)
    assert weights.shape == (20, 10)
