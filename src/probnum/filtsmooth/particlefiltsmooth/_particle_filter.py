"""Particle filters."""

import abc
from dataclasses import dataclass

import numpy as np

from probnum import random_variables
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior

from ._particle_posterior import ParticleFilterPosterior


def effective_number_of_events(categ_rv):
    return 1.0 / np.sum(categ_rv.event_probabilities ** 2)


class ParticleFilterState:
    """
    Set of weighted particles used within the particle filter.    Collection of :math:`N` particles :math:`(X_{ij})_{ij}`
    in :math:`m` dimensions
    with :math:`N` weights :math:`(w_i)_i`.    Attributes
    ---------
    particles : np.ndarray, shape=(N, m)
        These are the particles
    weights : np.ndarray, shape=(N,)
        These are the weights.
    """

    def __init__(self, particles, weights):
        self.particles = particles
        self.weights = weights

    @property
    def effective_num_particles(self):
        return 1.0 / np.sum(self.weights ** 2)


class ParticleFilter(BayesFiltSmooth):
    """Particle filter."""

    def __init__(
        self,
        dynamics_model,
        measurement_model,
        initrv,
        num_particles,
        importance_density_choice="gaussian",
        with_resampling=True,
    ):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv
        self.num_particles = num_particles
        self.with_resampling = with_resampling

        if importance_density_choice not in ["bootstrap", "gaussian"]:
            raise ValueError
        self.importance_density_choice = importance_density_choice

    def filter(self, dataset, times, _previous_posterior=None):

        # Initialize

        particles = []
        weights = []
        for idx in range(self.num_particles):

            dynamics_rv = self.initrv
            data = dataset[0]
            t = times[0]
            proposal_rv = self.dynamics_to_proposal_rv(dynamics_rv, data, t)
            proposal_state = proposal_rv.sample()

            meas_rv, _ = self.measurement_model.forward_realization(
                proposal_state, t=times[0]
            )

            proposal_weight = (
                meas_rv.pdf(dataset[0])
                * dynamics_rv.pdf(proposal_state)
                / proposal_rv.pdf(proposal_state)
            )

            particles.append(proposal_state)
            weights.append(proposal_weight)

        weights = weights / np.sum(weights)

        new_particle_state = ParticleFilterState(
            particles=np.array(particles), weights=weights
        )
        states = [new_particle_state]

        # Iterate

        for idx in range(1, len(times)):

            new_particle_state, _ = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                randvar=new_particle_state,
                data=dataset[idx],
            )
            states.append(new_particle_state)
        return ParticleFilterPosterior(states, times)

    def dynamics_to_proposal_rv(self, dynamics_rv, data, t):
        proposal_rv = dynamics_rv
        if self.importance_density_choice == "gaussian":  # "gaussian"
            proposal_rv, _ = self.measurement_model.backward_realization(
                data, proposal_rv, t=t
            )
        return proposal_rv

    def filter_step(self, start, stop, randvar, data):
        new_particle_state = ParticleFilterState(
            particles=randvar.particles.copy(),
            weights=randvar.weights.copy(),
        )

        for idx, (particle, weight) in enumerate(
            zip(new_particle_state.particles, new_particle_state.weights)
        ):

            dynamics_rv, _ = self.dynamics_model.forward_realization(
                particle, t=start, dt=(stop - start)
            )

            proposal_rv = self.dynamics_to_proposal_rv(dynamics_rv, data=data, t=stop)
            proposal_state = proposal_rv.sample()

            meas_rv, _ = self.measurement_model.forward_realization(
                proposal_state, t=stop
            )

            proposal_weight = (
                meas_rv.pdf(data)
                * dynamics_rv.pdf(proposal_state)
                / proposal_rv.pdf(proposal_state)
            )

            new_particle_state.particles[idx] = proposal_state
            new_particle_state.weights[idx] = proposal_weight

        new_particle_state.weights = new_particle_state.weights / np.sum(
            new_particle_state.weights
        )

        # Resample
        if self.with_resampling:
            if new_particle_state.effective_num_particles < self.num_particles / 10:
                print("Resampling")
                assert False
                u = np.random.rand(len(new_particle_state.weights))
                bins = np.cumsum(new_particle_state.weights)
                new_particle_state = ParticleFilterState(
                    particles=new_particle_state.particles[np.digitize(u, bins)],
                    weights=np.ones(len(new_particle_state.weights))
                    / len(new_particle_state.weights),
                )

        return new_particle_state, {}
