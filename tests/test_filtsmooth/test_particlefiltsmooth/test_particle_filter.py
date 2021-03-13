import matplotlib.pyplot as plt
import numpy as np
import pytest

from probnum import filtsmooth, random_variables, statespace

from ..filtsmooth_testcases import pendulum


def test_effective_number_of_events():
    weights = np.random.rand(10)
    categ = random_variables.Categorical(
        support=np.random.rand(10, 2), event_probabilities=weights / np.sum(weights)
    )
    ess = filtsmooth.effective_number_of_events(categ)
    assert 0 < ess < 10


@pytest.fixture
def num_particles():
    return 5


@pytest.fixture
def pendulum_problem():
    return pendulum()


@pytest.fixture
def data(pendulum_problem):
    dynamod, measmod, initrv, info = pendulum_problem
    delta_t = info["dt"]
    tmax = info["tmax"]
    times = np.arange(0, tmax, delta_t)
    states, obs = statespace.generate_samples(dynamod, measmod, initrv, times)

    # Introduce clutter
    # for idx in range(len(obs) // 2):
    #     obs[2 * idx] = 4 * np.random.rand() - 2
    return states, obs, times
    #
    # locations = np.linspace(0, 2 * np.pi, num_gridpoints)
    # data = 0.005 * np.random.randn(num_gridpoints) + np.sin(locations)
    # return data, locations


@pytest.fixture
def setup(pendulum_problem, num_particles):
    dynmod, measmod, initrv, info = pendulum_problem
    linearized_measmod = filtsmooth.DiscreteUKFComponent(measmod)

    particle = filtsmooth.ParticleFilter(
        dynmod,
        measmod,
        initrv,
        num_particles=num_particles,
        linearized_measurement_model=None,
    )
    return dynmod, measmod, initrv, particle


def test_sth(setup, data, num_particles):
    true_states, obs, locations = data
    prior, measmod, initrv, particle = setup

    posterior = particle.filter(obs.reshape((-1, 1)), locations)
    states = posterior.supports
    weights = posterior.event_probabilities

    num_gridpoints = len(locations)
    assert states.shape == (num_gridpoints, num_particles, 2)
    assert weights.shape == (num_gridpoints, num_particles)

    mean = posterior.mean
    mode = posterior.mode
    # cov = posterior.cov
    print(np.linalg.norm(mean - true_states) / np.sqrt(true_states.size))
    # print(cov.shape)

    # for i in range(num_particles):
    #     for l, p, w in zip(locations, states[:, i, 0], weights[:, i]):
    #
    #         plt.plot(l, np.sin(p), "o", alpha=min(0.0 + w, 1.0), color="k")

    plt.plot(locations, np.sin(true_states[:, 0]), label="states)")
    plt.plot(locations, np.sin(mean[:, 0]), label="mean(posterior)")
    plt.plot(locations, np.sin(mode[:, 0]), label="mode(posterior)")
    plt.plot(locations, obs, label="Observations", marker="x", linestyle="None")
    plt.legend()
    # plt.ylim((-2, 2))
    plt.show()

    assert False
