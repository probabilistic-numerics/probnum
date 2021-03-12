"""Particle filters."""

import abc
from dataclasses import dataclass

import numpy as np

from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior

from ._particle_posterior import ParticleFilterPosterior


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
        importance_density=None,
    ):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv
        self.num_particles = num_particles

        # Fallback: bootstrap particle filter
        self.importance_distribution = (
            importance_density if importance_density is not None else dynamics_model
        )

    def filter(self, dataset, times, _previous_posterior=None):

        # Initialize

        new_particle_state = self.initialize()
        for idx, (particle, weight) in enumerate(
            zip(new_particle_state.particles, new_particle_state.weights)
        ):
            proposal_state = particle
            proposal_rv = self.initrv
            dynamics_rv = self.initrv

            meas_rv, _ = self.measurement_model.forward_realization(
                proposal_state, t=times[0]
            )

            proposal_weight = (
                meas_rv.pdf(dataset[0])
                * dynamics_rv.pdf(proposal_state)
                / proposal_rv.pdf(proposal_state)
            )

            new_particle_state.particles[idx] = proposal_state
            new_particle_state.weights[idx] = proposal_weight

        new_particle_state.weights = new_particle_state.weights / np.sum(
            new_particle_state.weights
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

    def initialize(self):
        particles = self.initrv.sample(size=self.num_particles)
        weights = np.ones(self.num_particles) / self.num_particles
        return ParticleFilterState(weights=weights, particles=particles)

    def filter_step(self, start, stop, randvar, data):
        particle_state = randvar
        new_particle_state = ParticleFilterState(
            particles=particle_state.particles.copy(),
            weights=particle_state.weights.copy(),
        )

        for idx, (particle, weight) in enumerate(
            zip(new_particle_state.particles, new_particle_state.weights)
        ):
            proposal_rv, _ = self.importance_distribution.forward_realization(
                particle, t=start, dt=(stop - start)
            )
            proposal_state = proposal_rv.sample()

            dynamics_rv, _ = self.dynamics_model.forward_realization(
                particle, t=start, dt=(stop - start)
            )
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

        return new_particle_state, {}
